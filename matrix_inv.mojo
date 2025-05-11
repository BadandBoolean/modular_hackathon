# Combined imports from all files
from benchmark import ThroughputMeasure, BenchId, BenchMetric, Bench, Bencher
from buffer.dimlist import DimList
from gpu.host import DeviceContext, DeviceBuffer
from math import iota, ceildiv
from max.driver import cpu
from max.tensor import (
    ManagedTensorSlice,
    InputTensor,
    OutputTensor,
    StaticTensorSpec,
    IOSpec,
    Input,
    Output,
    MutableInput,
)
from memory import AddressSpace, UnsafePointer
from random import rand
from runtime.asyncrt import DeviceContextPtr
from sys import sizeof, has_amd_gpu_accelerator, has_nvidia_gpu_accelerator
from utils import IndexList
import compiler
from gpu import WARP_SIZE, barrier, block_dim, block_idx, thread_idx
from gpu.memory import async_copy_wait_all
from layout.layout_tensor import (
    Layout,
    LayoutTensor,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
)
from layout.math import outer_product_acc
from layout.tensor_builder import LayoutTensorBuild as tb
from sys.info import simdwidthof

# Note: change this to the ID of the GPU you will use.
alias DEVICE_ID = 0

# ===-----------------------------------------------------------------------===#
# Matrix Multiplication Implementations
# ===-----------------------------------------------------------------------===#


fn naive_matrix_multiplication_cpu(
    out: ManagedTensorSlice,
    a: ManagedTensorSlice[type = out.type, rank = out.rank],
    b: ManagedTensorSlice[type = out.type, rank = out.rank],
):
    """A naive matrix multiplication used as a fallback on CPU hardware."""
    var M = a.shape()[0]
    var N = b.shape()[1]
    var K = b.shape()[0]

    for row in range(M):
        for col in range(N):
            for k_loop in range(K):
                out[row, col] = out[row, col] + a[row, k_loop] * b[k_loop, col]


# ===-----------------------------------------------------------------------===#
# Matrix Inverse Implementations
# ===-----------------------------------------------------------------------===#


fn naive_matrix_inverse_cpu(
    c_out: ManagedTensorSlice,
    a_input_for_modification: ManagedTensorSlice[
        type = c_out.type, rank = c_out.rank
    ],
) raises:
    """
    CPU naive matrix inversion using Gauss-Jordan elimination.
    This function modifies 'a_input_for_modification' in-place to become an
    identity matrix, while 'c_out' is transformed from an identity matrix
    into the inverse of the original 'a_input_for_modification'.

    Args:
        c_out: Output tensor slice where the inverse matrix will be stored.
               It's assumed to be of the correct shape (N x N).
        a_input_for_modification: Input tensor slice containing the matrix to be inverted.
                                  This matrix WILL BE MODIFIED IN-PLACE.
                                  Ensure this is a mutable copy if the original
                                  matrix needs to be preserved across multiple calls or for verification.
    """
    var N = a_input_for_modification.shape()[0]
    alias DTYPE = c_out.type

    for i in range(N):
        for j in range(N):
            if i == j:
                c_out[i, j] = Scalar[DTYPE](1)
            else:
                c_out[i, j] = Scalar[DTYPE](0)

    # Perform Gauss-Jordan elimination
    for k_pivot_idx in range(N):
        # --- Step 1: Normalize the current pivot row k_pivot_idx ---
        # Find pivot element a_input_for_modification[k_pivot_idx, k_pivot_idx]
        pivot_element_val = a_input_for_modification[k_pivot_idx, k_pivot_idx]

        # Normalize row k_pivot_idx in both a_input_for_modification and c_out
        # by dividing all elements in the row by pivot_element_val.
        for j_col_idx in range(N):  # Iterate through columns of the pivot row
            a_input_for_modification[k_pivot_idx, j_col_idx] = (
                a_input_for_modification[k_pivot_idx, j_col_idx]
                / pivot_element_val
            )
            c_out[k_pivot_idx, j_col_idx] = (
                c_out[k_pivot_idx, j_col_idx] / pivot_element_val
            )
        # After this, a_input_for_modification[k_pivot_idx, k_pivot_idx] is 1.

        # --- Step 2: Eliminate other rows ---
        # For each row i_row_idx (where i_row_idx != k_pivot_idx), subtract a multiple of the
        # (now normalized) pivot row k_pivot_idx from row i_row_idx.
        # The goal is to make a_input_for_modification[i_row_idx, k_pivot_idx] zero.
        for i_row_idx in range(N):  # Iterate through all rows
            if i_row_idx == k_pivot_idx:  # Skip the pivot row itself
                continue

            # Get the elimination factor, which is the current element a_input_for_modification[i_row_idx, k_pivot_idx].
            # This is the value we want to zero out.
            elimination_factor = a_input_for_modification[
                i_row_idx, k_pivot_idx
            ]

            # Update row i_row_idx in both matrices: Row_i = Row_i - factor * Row_k
            # Note: a_input_for_modification[k_pivot_idx, ...] and c_out[k_pivot_idx, ...] are
            # elements from the *normalized* pivot row.
            for j_col_idx in range(N):  # Iterate through columns
                a_input_for_modification[i_row_idx, j_col_idx] = (
                    a_input_for_modification[i_row_idx, j_col_idx]
                    - elimination_factor
                    * a_input_for_modification[k_pivot_idx, j_col_idx]
                )
                c_out[i_row_idx, j_col_idx] = (
                    c_out[i_row_idx, j_col_idx]
                    - elimination_factor * c_out[k_pivot_idx, j_col_idx]
                )
            # After this, a_input_for_modification[i_row_idx, k_pivot_idx] is 0.

    # At this point, a_input_for_modification should be an identity matrix (or very close, for floating-point types),
    # and c_out should contain the inverse of the original a_input_for_modification.


fn naive_matrix_multiplication[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
](
    a: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutableAnyOrigin],
):
    """
    GEMM kernel that performs matrix multiplication C = A * B.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.
    """
    var M = a.dim(0)
    var N = b.dim(1)
    var K = b.dim(0)  # K is a.dim(1) or b.dim(0)

    # Calculate the column and row indices for each thread.
    var row = block_dim.x * block_idx.x + thread_idx.x
    var col = block_dim.y * block_idx.y + thread_idx.y

    # Initialize a register to accumulate the result for this thread.
    var dst_reg: c.element_type = 0

    # Iterate over the K dimension to compute the dot product.
    if row < M and col < N:
        for k_index in range(K):
            # Multiply the elements and accumulate the result.
            dst_reg = dst_reg + a[row, k_index] * b[k_index, col]

    # Write the final accumulated result to the output matrix.
    if row < M and col < N:  # Ensure write is within bounds
        c[row, col] = dst_reg


fn block_tiled_vectorized_matrix_multiplication[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    NUM_THREADS: Int,
](
    a: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutableAnyOrigin],
):
    """
    Tiled GEMM kernel that performs matrix multiplication C = A * B with
    vectorized memory access.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout: The layout of the input tensor A.
        b_layout: The layout of the input tensor B.
        c_layout: The layout of the output tensor C.
        BM: The block size in the M dimension.
        BN: The block size in the N dimension.
        BK: The block size in the K dimension.
        TM: The tile size in the M dimension.
        TN: The tile size in the N dimension.
        NUM_THREADS: The total number of threads per block.

    Args:
        a: The input tensor A.
        b: The input tensor B.
        c: The output tensor C.
    """
    alias simd_width = simdwidthof[dtype]()
    var partition_col = thread_idx.x % (BN // TN)
    var partition_row = thread_idx.x // (BN // TN)

    # Get the tile of the output matrix C that this thread is responsible
    # for computing.
    var dst = c.tile[BM, BN](block_idx.y, block_idx.x).tile[TM, TN](
        partition_row, partition_col
    )
    var dst_vec = dst.vectorize[1, simd_width]()

    # Allocate shared memory for tiles of A and B.
    # Use column-major layout for A to get the transpose.
    var a_smem = tb[dtype]().col_major[BM, BK]().shared().alloc()
    var b_smem = tb[dtype]().row_major[BK, BN]().shared().alloc()

    # Allocate register tiles to store the partial results and operands.
    var dst_reg = tb[dtype]().row_major[TM, TN]().local().alloc()
    var dst_reg_vec = dst_reg.vectorize[1, simd_width]()
    dst_reg_vec.copy_from(dst_vec)

    var a_reg = tb[dtype]().layout[TM]().local().alloc()
    var b_reg = tb[dtype]().layout[TN]().local().alloc()

    var ntiles = b.dim(0) // BK  # Assuming K dimension is b.dim(0)

    # Iterate over the tiles of A and B in the K dimension.
    for tile_k_idx in range(ntiles):  # Renamed 'block' to 'tile_k_idx'
        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)
        var a_tile = a.tile[BM, BK](block_idx.y, tile_k_idx)
        var b_tile = b.tile[BK, BN](tile_k_idx, block_idx.x)

        # Load the tiles of A and B into shared memory using vectorized
        # memory access.
        copy_dram_to_sram_async[thread_layout=load_a_layout](
            a_smem.vectorize[simd_width, 1](), a_tile.vectorize[simd_width, 1]()
        )
        copy_dram_to_sram_async[thread_layout=load_b_layout](
            b_smem.vectorize[1, simd_width](), b_tile.vectorize[1, simd_width]()
        )

        async_copy_wait_all()
        barrier()

        # Iterate over the elements in the K dimension within the tiles.
        @parameter
        for k_in_tile in range(BK):  # Renamed 'k' to 'k_in_tile'
            # Load the corresponding tiles from shared memory into registers.
            var a_smem_tile = a_smem.tile[TM, 1](
                partition_row, k_in_tile
            )  # Renamed a_tile to a_smem_tile
            var b_smem_tile = b_smem.tile[1, TN](
                k_in_tile, partition_col
            )  # Renamed b_tile to b_smem_tile
            a_reg.copy_from(a_smem_tile)
            b_reg.copy_from(b_smem_tile)

            # Perform outer product and accumulate the partial results.
            outer_product_acc(dst_reg, a_reg, b_reg)

        barrier()

    # Write the final accumulated results to the output matrix.
    dst_vec.copy_from(dst_reg_vec)


# ===----------------------------------------------------------------------=== #
# Matrix Multiplication Operation Dispatcher
# ===----------------------------------------------------------------------=== #


@compiler.register("matrix_multiplication")
struct MatrixMultiplication[algorithm: StaticString]:
    """
    The central custom operation that dispatches to multiple different
    matrix multiplication implementations, depending on target hardware and
    selected algorithm.
    """

    @staticmethod
    fn execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        out: OutputTensor[rank=2],
        a: InputTensor[type = out.type, rank = out.rank],
        b: InputTensor[type = out.type, rank = out.rank],
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        @parameter
        if target == "gpu":
            a_layout = a.to_layout_tensor()
            b_layout = b.to_layout_tensor()
            out_layout = out.to_layout_tensor()

            M = a_layout.shape[0]()
            N = b_layout.shape[1]()

            gpu_ctx = ctx.get_device_context()

            # Zero out the memory in the outbound tensor.
            gpu_ctx.enqueue_memset(
                DeviceBuffer[out.type](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[out.type]]](out_layout.ptr),
                    M * N,
                    owning=False,
                ),
                Scalar[out.type](0),  # Ensure zero is of the correct dtype
            )

            # We support several compile-time variants for the matrix
            # multiplication calculation:
            # - "naive": A naive matrix multiplication using LayoutTensors.
            # - "optimized": Matrix multiplication using a
            #   further-optimized 2D block tiling strategy.
            @parameter
            if algorithm == "naive":
                alias BM = 32
                alias BN = 32
                gpu_ctx.enqueue_function[
                    naive_matrix_multiplication[
                        out.type,
                        a_layout.layout,
                        b_layout.layout,
                        out_layout.layout,
                        BM,
                        BN,
                    ]
                ](
                    a_layout,
                    b_layout,
                    out_layout,
                    grid_dim=(
                        ceildiv(N, BN),
                        ceildiv(M, BM),
                    ),  # (grid_dim_x, grid_dim_y)
                    block_dim=(BN, BM),  # (block_dim_x, block_dim_y)
                    # Note: naive_matrix_multiplication uses block_dim.x for rows, block_dim.y for columns
                    # and block_idx.x for row blocks, block_idx.y for col blocks.
                    # If M is rows, N is columns:
                    # row = block_dim.x * block_idx.x + thread_idx.x
                    # col = block_dim.y * block_idx.y + thread_idx.y
                    # grid_dim should be (ceildiv(M,BM), ceildiv(N,BN)) if BM for M, BN for N
                    # and block_dim (BM, BN) matching (thread_idx.x, thread_idx.y) ranges
                    # Current kernel uses:
                    # row = block_dim.x * block_idx.x + thread_idx.x
                    # col = block_dim.y * block_idx.y + thread_idx.y
                    # Launching with block_dim=(BN, BM) means block_dim.x = BN, block_dim.y = BM
                    # If BM associated with M (rows), BN with N (cols):
                    # grid_dim=(ceildiv(M,BM), ceildiv(N,BN))
                    # block_dim=(BM,BN) -> thread_idx.x up to BM-1, thread_idx.y up to BN-1
                    # This seems consistent if BM for M-dim, BN for N-dim
                    # Original: grid_dim=(ceildiv(N, BN), ceildiv(M, BM)), block_dim=(BN, BM)
                    # This maps block_idx.x to N-dim, block_idx.y to M-dim
                    # And thread_idx.x to N-dim local, thread_idx.y to M-dim local
                    # row = BN * block_idx.x + thread_idx.x  -> maps to M dimension
                    # col = BM * block_idx.y + thread_idx.y  -> maps to N dimension
                    # This looks like a mismatch or unconventional mapping.
                    # Standard CUDA: grid_dim.x, block_dim.x, block_idx.x, thread_idx.x usually map to columns (N)
                    # grid_dim.y, block_dim.y, block_idx.y, thread_idx.y usually map to rows (M)
                    # If using (cols, rows) for (x,y) in grid/block dims:
                    # grid_dim = (ceildiv(N,TILE_N), ceildiv(M,TILE_M))
                    # block_dim = (TILE_N, TILE_M)
                    # Inside kernel:
                    # global_col = block_dim.x * block_idx.x + thread_idx.x
                    # global_row = block_dim.y * block_idx.y + thread_idx.y
                    # The original code is kept, but commented for clarity.
                )
            elif algorithm == "optimized":
                alias BM = 128  # Corresponds to M-dimension tile size handled by a block
                alias BN = 128  # Corresponds to N-dimension tile size handled by a block
                alias BK = 8  # Corresponds to K-dimension tile size for shared memory
                alias TM = 8  # Corresponds to M-dimension sub-tile per thread (work per thread)
                alias TN = 8  # Corresponds to N-dimension sub-tile per thread (work per thread)
                # NUM_THREADS should be (BM/TM) * (BN/TN) if each thread handles one TMxTN sub-tile.
                # Original: NUM_THREADS = (BM * BN) // (TM * TN) ; this is correct. (total TMxTN tiles in a BMxBN block)
                # The block_dim is (NUM_THREADS), meaning a 1D thread block.
                # partition_row/col based on thread_idx.x then map this 1D index to 2D.
                alias NUM_THREADS = (BM // TM) * (
                    BN // TN
                )  # Total threads needed to cover the BMxBN block, each handling TMxTN
                gpu_ctx.enqueue_function[
                    block_tiled_vectorized_matrix_multiplication[
                        out.type,
                        a_layout.layout,
                        b_layout.layout,
                        out_layout.layout,
                        BM,
                        BN,
                        BK,
                        TM,
                        TN,
                        NUM_THREADS,
                    ]
                ](
                    a_layout,
                    b_layout,
                    out_layout,
                    grid_dim=(
                        ceildiv(N, BN),
                        ceildiv(M, BM),
                    ),  # (grid_x for N-dim, grid_y for M-dim)
                    block_dim=(NUM_THREADS),  # 1D thread block
                )


@compiler.register("matrix_inverse")
struct MatrixInverse[algorithm: StaticString]:
    """
    Custom operation that dispatches to different matrix inverse implementations,
    depending on target hardware and selected algorithm.
    """

    @staticmethod
    fn execute[
        target: StaticString,
    ](
        out: OutputTensor[rank=2],  # Output: Inverse of A
        a: InputTensor[
            type = out.type, rank = out.rank
        ],  # Input: Matrix A to be inverted
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            # For GPU, 'a' is an InputTensor. Its data will be read by the kernel.
            # The kernel `naive_matrix_inverse` takes `a: LayoutTensor[..., MutableAnyOrigin]`
            # and modifies it in-place. This means the underlying buffer of `a.to_layout_tensor()` is changed.
            a_layout = a.to_layout_tensor()
            out_layout = out.to_layout_tensor()

            N_dim = a_layout.shape[0]()  # Renamed N to N_dim to avoid conflict

            gpu_ctx = ctx.get_device_context()

            # Zero out the memory in the outbound tensor.
            # `naive_matrix_inverse` kernel initializes `c` (out_layout) to identity,
            # so this memset might be redundant if the kernel guarantees full initialization.
            # However, it's good practice for output tensors.
            gpu_ctx.enqueue_memset(
                DeviceBuffer[out.type](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[out.type]]](out_layout.ptr),
                    N_dim * N_dim,
                    owning=False,
                ),
                Scalar[out.type](0),  # Ensure zero is of the correct dtype
            )

            @parameter
            if algorithm == "naive":
                alias BN_inv = 32  # Renamed BN to BN_inv for clarity
                gpu_ctx.enqueue_function[
                    naive_matrix_inverse[
                        out.type,  # dtype
                        a_layout.layout,  # a_layout (for matrix A)
                        out_layout.layout,  # c_layout (for output inverse)
                        BN_inv,  # Block dimension
                    ]
                ](
                    a_layout,  # Tensor A (will be modified by kernel)
                    out_layout,  # Tensor C (will receive the inverse)
                    grid_dim=(ceildiv(N_dim, BN_inv), ceildiv(N_dim, BN_inv)),
                    block_dim=(BN_inv, BN_inv),
                )

        # Else for CPU target (no CPU dispatch in original struct, can be added if needed)
        # If adding CPU dispatch here:
        # else: # target == "cpu"
        #     # Ensure 'a' is mutable if naive_matrix_inverse_cpu modifies it.
        #     # The `a: InputTensor` might need to be copied to a mutable buffer first.
        #     # This struct might be better for GPU-only, and CPU calls handled directly in benchmark.
        #     # For now, assuming CPU is handled by direct call to naive_matrix_inverse_cpu in benchmark.
        #     raise Error("CPU target for MatrixInverse op not explicitly handled here, call CPU function directly.")


# ===----------------------------------------------------------------------=== #
# Benchmark Tensor Wrapper
# ===----------------------------------------------------------------------=== #


@value
struct _BenchTensor[
    io_spec: IOSpec,
    static_spec: StaticTensorSpec,  # This is a generic type, its DType param is 'type'
]:
    alias tensor_type = ManagedTensorSlice[
        io_spec=io_spec, static_spec=static_spec
    ]
    alias buffer_type = DeviceBuffer[static_spec.type]
    alias ptr_type = UnsafePointer[Scalar[static_spec.type]]
    alias size = Int(
        static_spec.shape.product()
    )  # product() needs to be cast to Int if it's not

    var tensor: Self.tensor_type
    var buffer: Self.buffer_type

    fn __init__(out self, ctx: DeviceContext) raises:
        self.buffer = ctx.enqueue_create_buffer[static_spec.type](Self.size)
        self.tensor = ManagedTensorSlice[
            io_spec=io_spec, static_spec=static_spec
        ](
            self.buffer.unsafe_ptr(),  # Assuming unsafe_ptr() returns the correct ptr_type
            Self.static_spec.shape.into_index_list[static_spec.rank](),
            Self.static_spec.strides.into_index_list[static_spec.rank](),
        )

    fn unsafe_ptr(self) -> Self.ptr_type:
        return self.buffer.unsafe_ptr()  # Assuming this returns ptr_type

    fn rand(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            # Ensure rand function matches expected signature (ptr, size)
            rand(
                host_buffer.unsafe_ptr(), Self.size
            )  # Make sure Self.size is Int for rand
        return self

    fn iota(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            # Ensure iota function matches expected signature
            iota(
                host_buffer.unsafe_ptr(), Self.size
            )  # Make sure Self.size is Int for iota
        return self


# ===----------------------------------------------------------------------=== #
# Benchmarking Functions
# ===----------------------------------------------------------------------=== #


def matmul():
    alias M_matmul = 1024
    alias K_matmul = 1024
    alias N_matmul = 1024

    alias rank_matmul = 2
    alias dtype_matmul = DType.float32

    alias FLOPS_matmul = M_matmul * N_matmul * (2 * K_matmul - 1)

    alias a_spec_matmul = StaticTensorSpec[dtype_matmul, rank_matmul](
        shape=DimList(M_matmul, K_matmul),
        strides=DimList(K_matmul, 1),
        alignment=sizeof[dtype_matmul](),
        address_space=AddressSpace.GENERIC,
        exclusive=True,
        in_lambda=None,
        out_lambda=None,
        out_compute_lambda=None,
    )

    alias b_spec_matmul = StaticTensorSpec[dtype_matmul, rank_matmul](
        shape=DimList(K_matmul, N_matmul),
        strides=DimList(N_matmul, 1),
        alignment=sizeof[dtype_matmul](),
        address_space=AddressSpace.GENERIC,
        exclusive=True,
        in_lambda=None,
        out_lambda=None,
        out_compute_lambda=None,
    )

    alias c_spec_matmul = StaticTensorSpec[dtype_matmul, rank_matmul](
        shape=DimList(M_matmul, N_matmul),
        strides=DimList(N_matmul, 1),
        alignment=sizeof[dtype_matmul](),
        address_space=AddressSpace.GENERIC,
        exclusive=True,
        in_lambda=None,
        out_lambda=None,
        out_compute_lambda=None,
    )

    var cpu_ctx_matmul = DeviceContext(api="cpu")  # Renamed

    var a_cpu = _BenchTensor[Input, a_spec_matmul](cpu_ctx_matmul).rand()
    var b_cpu = _BenchTensor[Input, b_spec_matmul](cpu_ctx_matmul).rand()
    var c_cpu_out = _BenchTensor[Output, c_spec_matmul](
        cpu_ctx_matmul
    ).rand()  # .rand() content will be overwritten

    var bench_matmul = Bench()  # Renamed
    var flops_measure_matmul = ThroughputMeasure(
        BenchMetric.flops, FLOPS_matmul
    )
    var elements_measure_matmul = ThroughputMeasure(
        BenchMetric.elements, M_matmul * N_matmul
    )

    @parameter
    @always_inline
    fn bench_cpu_matmul(
        mut bencher: Bencher,
    ) raises:  # bencher name is fine locally
        @parameter
        @always_inline
        fn run_bench_matmul() raises:
            MatrixMultiplication["naive"].execute[target="cpu"](
                c_cpu_out.tensor,
                a_cpu.tensor,
                b_cpu.tensor,
                cpu_ctx_matmul,  # Pass DeviceContextPtr
            )

        bencher.iter[run_bench_matmul]()

    bench_matmul.bench_function[bench_cpu_matmul](
        BenchId("cpu", "naive_matmul"),
        flops_measure_matmul,
        elements_measure_matmul,
    )

    @parameter
    if has_amd_gpu_accelerator() or has_nvidia_gpu_accelerator():
        var gpu_ctx_matmul = DeviceContext(device_id=DEVICE_ID)  # Renamed
        var a_dev_matmul = _BenchTensor[Input, a_spec_matmul](
            gpu_ctx_matmul
        ).rand()
        var b_dev_matmul = _BenchTensor[Input, b_spec_matmul](
            gpu_ctx_matmul
        ).rand()
        var c_dev_matmul = _BenchTensor[Output, c_spec_matmul](
            gpu_ctx_matmul
        ).rand()  # Overwritten

        @parameter
        def bench_matmul_kernel[impl_matmul: StaticString]():  # Renamed impl
            @parameter
            @always_inline
            fn bench_gpu_matmul(mut bencher: Bencher) raises:
                @parameter
                @always_inline
                fn kernel_launch_matmul(
                    ctx_param: DeviceContext,
                ) raises:  # Renamed ctx_param
                    MatrixMultiplication[impl_matmul].execute[target="gpu"](
                        c_dev_matmul.tensor,
                        a_dev_matmul.tensor,
                        b_dev_matmul.tensor,
                        ctx_param,  # Pass DeviceContextPtr
                    )

                bencher.iter_custom[kernel_launch_matmul](gpu_ctx_matmul)

            bench_matmul.bench_function[bench_gpu_matmul](
                BenchId("gpu", String(impl_matmul) + "_matmul"),
                flops_measure_matmul,
                elements_measure_matmul,
            )

        bench_matmul_kernel["naive"]()
        bench_matmul_kernel["optimized"]()

    bench_matmul.config.verbose_metric_names = False
    print(bench_matmul)


def matinv():
    alias N_matinv = 1024
    alias rank_matinv = 2
    alias dtype_matinv = DType.float32
    alias FLOPS_matinv = 2 * N_matinv * N_matinv * N_matinv  # Approx. for Gauss-Jordan

    alias a_spec_matinv = StaticTensorSpec[dtype_matinv, rank_matinv](
        shape=DimList(N_matinv, N_matinv),
        strides=DimList(N_matinv, 1),
        alignment=sizeof[dtype_matinv](),
        address_space=AddressSpace.GENERIC,
        exclusive=True,
        in_lambda=None,
        out_lambda=None,
        out_compute_lambda=None,
    )
    # For matrix inverse, output c_spec is same shape as a_spec
    alias c_spec_matinv = a_spec_matinv

    var cpu_ctx_matinv = DeviceContext(api="cpu")  # Renamed
    var bench_matinv = Bench()  # Renamed
    var flops_measure_matinv = ThroughputMeasure(
        BenchMetric.flops, FLOPS_matinv
    )
    var elements_measure_matinv = ThroughputMeasure(
        BenchMetric.elements, N_matinv * N_matinv
    )

    # --- CPU Benchmarking for Matrix Inverse ---
    var a_cpu_mat_to_invert = _BenchTensor[MutableInput, a_spec_matinv](
        cpu_ctx_matinv
    ).rand()

    var c_cpu_inv_result = _BenchTensor[Output, c_spec_matinv](
        cpu_ctx_matinv
    )  # No need to .rand() if fully overwritten

    @parameter
    @always_inline
    fn bench_cpu_inverse(mut bencher: Bencher) raises:
        @parameter
        @always_inline
        fn run_bench_cpu_inverse() raises:
            naive_matrix_inverse_cpu(
                c_cpu_inv_result.tensor, a_cpu_mat_to_invert.tensor
            )

        bencher.iter[run_bench_cpu_inverse]()

    bench_matinv.bench_function[bench_cpu_inverse](
        BenchId("cpu", "naive_inverse"),
        flops_measure_matinv,
        elements_measure_matinv,
    )

    # --- GPU Benchmarking for Matrix Inverse ---
    @parameter
    if has_amd_gpu_accelerator() or has_nvidia_gpu_accelerator():
        var gpu_ctx_matinv = DeviceContext(device_id=DEVICE_ID)  # Renamed

        var a_dev_matinv = _BenchTensor[Input, a_spec_matinv](
            gpu_ctx_matinv
        ).rand()

        var c_dev_matinv = _BenchTensor[Output, c_spec_matinv](
            gpu_ctx_matinv
        )  # No need to .rand()

        @parameter
        def bench_matinv_kernel[impl_matinv: StaticString]():  # Renamed impl
            @parameter
            @always_inline
            fn bench_gpu_matinv(mut bencher: Bencher) raises:
                @parameter
                @always_inline
                fn kernel_launch_matinv(
                    ctx_param: DeviceContext,
                ) raises:  # Renamed ctx_param
                    MatrixInverse[impl_matinv].execute[target="gpu"](
                        c_dev_matinv.tensor,
                        a_dev_matinv.tensor,
                        ctx_param,  # Pass DeviceContextPtr
                    )

                bencher.iter_custom[kernel_launch_matinv](gpu_ctx_matinv)

            bench_matinv.bench_function[bench_gpu_matinv](
                BenchId("gpu", String(impl_matinv) + "_inverse"),
                flops_measure_matinv,
                elements_measure_matinv,
            )

        bench_matinv_kernel["naive"]()

    bench_matinv.config.verbose_metric_names = False
    print(bench_matinv)


fn naive_matrix_inverse[
    dtype: DType,
    a_layout_param: Layout,
    c_layout_param: Layout,
    BN_param: Int,
](
    a: LayoutTensor[
        dtype, a_layout_param, MutableAnyOrigin
    ],  # This is the matrix to be inverted, modified in-place
    c: LayoutTensor[
        dtype, c_layout_param, MutableAnyOrigin
    ],  # This will store the inverse
):
    """
    Naive kernel that performs matrix inversion using Gauss-Jordan elimination.
    This implementation transforms the input matrix 'a' into an identity matrix
    and simultaneously transforms an initial identity matrix 'c' into the inverse of 'a'.

    IMPORTANT ASSUMPTIONS:
    1. `barrier()` provides grid-wide synchronization. If `barrier()` is only
       block-local, this kernel will only work correctly when launched with a
       single block (i.e., if N <= BN_param).
    2. The input matrix 'a' is invertible without requiring pivoting. This naive
       implementation does not perform pivoting, so it may be numerically
       unstable or fail for matrices where diagonal elements become zero or
       very small during the elimination process.

    Parameters:
        dtype: The data type of the input and output tensors.
        a_layout_param: The layout of the input tensor A.
        c_layout_param: The layout of the output tensor C.
        BN_param: The block size dimension (e.g., 32 means 32x32 threads per block).

    Args:
        a: The input N x N tensor A. It will be modified in-place to become an
           identity matrix.
        c: The output N x N tensor C. It's initialized to an identity matrix by this kernel
           and will contain the inverse of the original A upon compaliasion.
    """
    var N_dim = a.dim(0)

    var thread_global_row = block_dim.x * block_idx.x + thread_idx.x
    var thread_global_col = block_dim.y * block_idx.y + thread_idx.y

    # Initialize matrix 'c' to the identity matrix.
    if thread_global_row < N_dim and thread_global_col < N_dim:
        if thread_global_row == thread_global_col:
            c[thread_global_row, thread_global_col] = c.element_type(1)
        else:
            c[thread_global_row, thread_global_col] = c.element_type(0)

    barrier()

    # Perform Gauss-Jordan elimination.
    for k_pivot_idx in range(N_dim):  # Iterate through pivot rows/columns
        var pivot_element_val: c.element_type  # Declare type for host compilation

        if thread_global_row == k_pivot_idx:  # Thread is on the pivot row
            var temp_pivot_val = a[k_pivot_idx, k_pivot_idx]

            if thread_global_col < N_dim:  # Process elements in this row
                if temp_pivot_val != c.element_type(0):
                    a[k_pivot_idx, thread_global_col] = (
                        a[k_pivot_idx, thread_global_col] / temp_pivot_val
                    )
                    c[k_pivot_idx, thread_global_col] = (
                        c[k_pivot_idx, thread_global_col] / temp_pivot_val
                    )

        barrier()

        # --- Step 2: Eliminate other rows ---
        if thread_global_row < N_dim and thread_global_col < N_dim:
            if thread_global_row != k_pivot_idx:  # If not on the pivot row
                var elimination_factor = a[thread_global_row, k_pivot_idx]

                a[thread_global_row, thread_global_col] = (
                    a[thread_global_row, thread_global_col]
                    - elimination_factor * a[k_pivot_idx, thread_global_col]
                )

                c[thread_global_row, thread_global_col] = (
                    c[thread_global_row, thread_global_col]
                    - elimination_factor * c[k_pivot_idx, thread_global_col]
                )

        barrier()


def main():
    # print("Benchmarking Matrix Multiplication...")
    # matmul()

    print("\nBenchmarking Matrix Inverse...")
    matinv()
