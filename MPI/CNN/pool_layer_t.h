#pragma once
#include "layer_t.h"
#include <mpi.h>

#pragma pack(push, 1)
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
	void activate()
	{
	    int rank, size;
	
	    // Initialize MPI environment
	    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	    // Divide the workload (output rows) among processes
	    int rows_per_process = out.size.x / size;
	    int start_row = rank * rows_per_process;
	    int end_row = (rank == size - 1) ? out.size.x : start_row + rows_per_process;
	
	    // Allocate memory for the local portion of the output
	    tensor_t<float> local_out(rows_per_process, out.size.y, out.size.z);
	
	    // Perform pooling for the assigned rows
	    for (int x = start_row; x < end_row; x++) {
	        for (int y = 0; y < out.size.y; y++) {
	            for (int z = 0; z < out.size.z; z++) {
	                point_t mapped = map_to_input({(uint16_t)x, (uint16_t)y, 0}, 0);
	                float mval = -FLT_MAX;
	
	                for (int i = 0; i < extend_filter; i++) {
	                    for (int j = 0; j < extend_filter; j++) {
	                        float v = in(mapped.x + i, mapped.y + j, z);
	                        if (v > mval) {
	                            mval = v;
	                        }
	                    }
	                }
	                local_out(x - start_row, y, z) = mval;
	            }
	        }
	    }
	
	    // Gather all local outputs into the root process
	    MPI_Gather(local_out.data, rows_per_process * out.size.y * out.size.z, MPI_FLOAT,
	               out.data, rows_per_process * out.size.y * out.size.z, MPI_FLOAT,
	               0, MPI_COMM_WORLD);
	
	  
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
