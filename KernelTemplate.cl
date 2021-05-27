

__kernel void GRAPH_DRAWING (__global char *image, __global int *dim, __global double *range)
{
	
	for (int j = get_global_id(0); j<dim[1]; j+=get_global_size(0))
	{
		float y = range[2] + (j + 0.5)*(range[3] - range[2]) / dim[1];
		for (int i = get_global_id(1) ; i<dim[0]; i+=get_global_size(1))
		{
			float x = range[0] + (i + 0.5)*(range[1] - range[0]) / dim[0];
			//m
			image[j*dim[0] + i] = (val>0) * 255;
		}
	}
}