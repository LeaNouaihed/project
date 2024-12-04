#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
__global__
void activate_kernel(const float* in, float* out, int in_x, int in_y, int in_z, int out_x, int out_y, int out_z, int stride, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // Output row index
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // Output column index
    int z = blockIdx.z;                            // Output depth (channel)

    // Ensure the thread is within bounds
    if (x < out_x && y < out_y && z < out_z) {
        int mapped_x = x * stride;                 // Map to input row
        int mapped_y = y * stride;                 // Map to input column
        float mval = -FLT_MAX;                     // Initialize max value for pooling

        // Perform max-pooling over the pooling window
        for (int i = 0; i < filter_size; i++) {
            for (int j = 0; j < filter_size; j++) {
                float val = in[(z * in_x * in_y) + ((mapped_y + j) * in_x) + (mapped_x + i)];
                if (val > mval) {
                    mval = val;                    // Update max value
                }
            }
        }

        // Store the maximum value in the output tensor
        out[(z * out_x * out_y) + (y * out_x) + x] = mval;
    }
}

struct pool_layer_t
{
	layer_type type = layer_type::pool;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	uint16_t stride;
	uint16_t extend_filter;

	pool_layer_t( uint16_t stride, uint16_t extend_filter, tdsize in_size )
		:
		grads_in( in_size.x, in_size.y, in_size.z ),
		in( in_size.x, in_size.y, in_size.z ),
		out(
		(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			in_size.z
		)

	{
		this->stride = stride;
		this->extend_filter = extend_filter;
		assert( (float( in_size.x - extend_filter ) / stride + 1)
				==
				((in_size.x - extend_filter) / stride + 1) );

		assert( (float( in_size.y - extend_filter ) / stride + 1)
				==
				((in_size.y - extend_filter) / stride + 1) );
	}

	point_t map_to_input( point_t out, int z )
	{
		out.x *= stride;
		out.y *= stride;
		out.z = z;
		return out;
	}

	struct range_t
	{
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
	};

	int normalize_range( float f, int max, bool lim_min )
	{
		if ( f <= 0 )
			return 0;
		max -= 1;
		if ( f >= max )
			return max;

		if ( lim_min ) // left side of inequality
			return ceil( f );
		else
			return floor( f );
	}

	range_t map_to_output( int x, int y )
	{
		float a = x;
		float b = y;
		return
		{
			normalize_range( (a - extend_filter + 1) / stride, out.size.x, true ),
			normalize_range( (b - extend_filter + 1) / stride, out.size.y, true ),
			0,
			normalize_range( a / stride, out.size.x, false ),
			normalize_range( b / stride, out.size.y, false ),
			(int)out.size.z - 1,
		};
	}

	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

	void activate() {
    // Allocate device memory
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, in.size.x * in.size.y * in.size.z * sizeof(float));
    cudaMalloc(&d_out, out.size.x * out.size.y * out.size.z * sizeof(float));

    // Copy input tensor to device
    cudaMemcpy(d_in, in.data, in.size.x * in.size.y * in.size.z * sizeof(float), cudaMemcpyHostToDevice);

    // Define CUDA grid and block configuration
    dim3 threads_per_block(16, 16);                // 16x16 threads per block
    dim3 num_blocks((out.size.x + threads_per_block.x - 1) / threads_per_block.x,
                    (out.size.y + threads_per_block.y - 1) / threads_per_block.y,
                    out.size.z);                   // 1 block per output channel

    // Launch the kernel
    activate_kernel<<<num_blocks, threads_per_block>>>(d_in, d_out,
                                                       in.size.x, in.size.y, in.size.z,
                                                       out.size.x, out.size.y, out.size.z,
                                                       stride, extend_filter);

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy output tensor back to host
    cudaMemcpy(out.data, d_out, out.size.x * out.size.y * out.size.z * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
}


	void fix_weights()
	{

	}

	void calc_grads( tensor_t<float>& grad_next_layer )
	{
		for ( int x = 0; x < in.size.x; x++ )
		{
			for ( int y = 0; y < in.size.y; y++ )
			{
				range_t rn = map_to_output( x, y );
				for ( int z = 0; z < in.size.z; z++ )
				{
					float sum_error = 0;
					for ( int i = rn.min_x; i <= rn.max_x; i++ )
					{
						int minx = i * stride;
						for ( int j = rn.min_y; j <= rn.max_y; j++ )
						{
							int miny = j * stride;

							int is_max = in( x, y, z ) == out( i, j, z ) ? 1 : 0;
							sum_error += is_max * grad_next_layer( i, j, z );
						}
					}
					grads_in( x, y, z ) = sum_error;
				}
			}
		}
	}
};
#pragma pack(pop)
