from sys import argv
from memory import memcpy, memset
from random import random_si64
from memory import UnsafePointer
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, print_layout
from sys.info import sizeof
from math import abs

alias dtype = DType.float32
alias SIZE = 3  # Change to desired size
alias layout = Layout.row_major(SIZE, SIZE)


# Find the pivot row with the largest absolute value in the current column
fn find_pivot(a: UnsafePointer[Scalar[dtype]], size: Int, col: Int) -> Int:
    max_idx = col
    max_val = abs(a[col * size + col])

    for i in range(col + 1, size):
        val = abs(a[i * size + col])
        if val > max_val:
            max_val = val
            max_idx = i

    return max_idx


# Swap two rows in the matrix
fn swap_rows(
    a: UnsafePointer[Scalar[dtype]], size: Int, row1: Int, row2: Int
) -> None:
    if row1 == row2:
        return

    for i in range(size):
        temp = a[row1 * size + i]
        a[row1 * size + i] = a[row2 * size + i]
        a[row2 * size + i] = temp


# Scale the pivot row
fn scale(
    a: UnsafePointer[Scalar[DType.float32]], size: Int, index: Int
) -> None:
    start = index * size + index
    end = index * size + size

    # Only scale if pivot is not zero
    if a[start] != 0:
        for i in range(start + 1, end):
            a[i] = a[i] / a[start]


# Reduce rows below the pivot
fn reduce(
    a: UnsafePointer[Scalar[DType.float32]], size: Int, index: Int, tid: Int
) -> None:
    start = (index + tid + 1) * size + index
    end = (index + tid + 1) * size + size

    # Store the multiplier for the L part first
    multiplier = a[start]

    # Then perform reduction on the remaining elements
    for i in range(start + 1, end):
        a[i] = a[i] - (multiplier * a[(index * size) + (index + (i - start))])


def main():
    ctx = DeviceContext()

    # Allocate memory
    a = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
    ctx.synchronize()

    # Initialize with random values
    for i in range(SIZE * SIZE):
        a[i] = Float32(random_si64(min=1, max=9))

    print("Input matrix:")
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            row.append(a[i * SIZE + j])
        print(row)

    # Create device buffers
    a_device_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
    ctx.synchronize()
    ctx.enqueue_copy(a_device_buf, a)
    ctx.synchronize()

    # Create row permutation vector to track pivoting
    perm = [i for i in range(SIZE)]

    # Wrap device buffer in LayoutTensor
    a_tensor = LayoutTensor[mut=True, dtype, layout](
        a_device_buf.unsafe_ptr()
    ).reshape[layout]()

    # Perform LU Decomposition with pivoting
    for i in range(SIZE - 1):  # Only need to process SIZE-1 rows
        # Find and apply pivot (bring the largest element to the diagonal)
        temp = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
        ctx.enqueue_copy(temp, a_device_buf)
        ctx.synchronize()

        pivot_row = find_pivot(temp.unsafe_ptr(), SIZE, i)

        if pivot_row != i:
            # Swap rows on host and update permutation vector
            swap_rows(temp.unsafe_ptr(), SIZE, i, pivot_row)
            # Update permutation tracking
            perm[i], perm[pivot_row] = perm[pivot_row], perm[i]
            # Copy back to device
            ctx.enqueue_copy(a_device_buf, temp)
            ctx.synchronize()

        # Scale the current row
        ctx.enqueue_function[scale](a_tensor, SIZE, i, grid_dim=1, block_dim=1)
        ctx.synchronize()

        # Reduce remaining rows (use one thread per row for better parallelism)
        if SIZE - i - 1 > 0:  # Only if there are rows remaining
            ctx.enqueue_function[reduce](
                a_tensor,
                SIZE,
                i,
                tid=0,
                grid_dim=1,
                block_dim=SIZE - i - 1,  # One thread per remaining row
            )
            ctx.synchronize()

    # Copy result back to host
    result = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
    ctx.enqueue_copy(result, a_device_buf)
    ctx.synchronize()

    print("Final LU Matrix:")
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            row.append(result[i * SIZE + j])
        print(row)

    print("Row permutation vector:", perm)
