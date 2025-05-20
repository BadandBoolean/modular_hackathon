from sys import argv
from memory import memcpy, memset
from random import random_si64
from memory import UnsafePointer
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, print_layout
from sys.info import sizeof

alias dtype = DType.float32
alias SIZE = 2
alias layout = Layout.row_major(SIZE, SIZE)


# divides all elements in the row by the diagonal (pivot) element
# executed in parallel on the specific row where the row is fed in but the
# element executions are parallelized
# this gets the upper triangular matrix
fn scale(
    a: UnsafePointer[Scalar[DType.float32]], size: Int, index: Int
) -> None:
    start = index * size + index
    end = index * size + size

    for i in range(start + 1, end):
        a[i] = a[i] / a[start]


# this gets the lower triangular matrix
# subtracts pivot el * multiplier from row
fn reduce(
    a: UnsafePointer[Scalar[DType.float32]], size: Int, index: Int, tid: Int
) -> None:
    start = (index + tid + 1) * size + index
    end = (index + tid + 1) * size + size
    for i in range(start + 1, end):
        a[i] = a[i] - (a[start] * a[(index * size) + (index + (i - start))])


def main():
    ctx = DeviceContext()

    # Allocate memory:
    a = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
    ctx.synchronize()

    # Initialize the input array
    for i in range(SIZE * SIZE):
        a[i] = Float32(random_si64(min=1, max=9))

    print("Input matrix:", a)
    # create device buffer for the input matrix
    a_device_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
    ctx.synchronize()

    # Copy for input from host to device
    ctx.enqueue_copy(a_device_buf, a)
    ctx.synchronize()
    print("Device input matrix:", a_device_buf)
    # Create device buffer for the output matrix
    result_device_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)

    # Wrap device buffers in LayoutTensor
    a_tensor = LayoutTensor[mut=True, dtype, layout](
        a_device_buf.unsafe_ptr()
    ).reshape[layout]()
    result_tensor = LayoutTensor[mut=True, dtype, layout](
        result_device_buf.unsafe_ptr()
    ).reshape[layout]()
    print_layout(layout)
    print("a_tensor initialized:", a_tensor)
    print("result_tensor initialized:", result_tensor)

    # Perform LU Decomposition
    for i in range(SIZE):
        # Scale the current row
        ctx.enqueue_function[scale](a_tensor, SIZE, i, grid_dim=1, block_dim=1)
        print("After scaling row", i, ":", result_tensor)
        ctx.synchronize()

        # Reduce remaining rows
        for tid in range(SIZE - i - 1):
            ctx.enqueue_function[reduce](
                a_tensor, SIZE, i, tid, grid_dim=1, block_dim=1
            )
            ctx.synchronize()

    # Create host buffer for output

    result_host_buf = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)

    # Copy result from device to host
    ctx.enqueue_copy(result_host_buf, a_device_buf)
    ctx.synchronize()
    print("Result:", result_host_buf)
