#include <stdio.h>
#include <cassert>
#include "legion.h"
using namespace std;
enum TaskIDS{
	TOP_LEVEL_TASK_ID,
	INIT_NUM_TASK_ID,
	SORT_TASK_ID,
};
enum FieldIDs{
	FID_X,
};
void top_level_task(const Task * task,const std::vector<PhysicalRegion> & regions,Context ctx,Runtime * runtime)
{
	int num_elements=1024;
	int num_subregions=1;
	printf("sort %d nums by %d partions\n",num_elements,num_subregions);
	Rect<1> elem_rect(0,num_elements-1);
	IndexSpace is=runtime->create_index_space(ctx,elem_rect);
	runtime->attach_name(is,"is");
	FieldSpace input_fs=runtime->create_field_space(ctx);
	runtime->attach_name(input_fs,"input_fs");
	{
		FieldAllocator allocator=runtime->create_field_allocator(ctx,input_fs);
		allocator.allocate_field(sizeof(double),FID_X);
		runtime->attach_name(input_fs,FID_X,"X");
	}
	LogicalRegion input_lr=runtime->create_logical_region(ctx,is,input_fs);
	runtime->attach_name(input_lr,"input_lr");
	Rect<1> color_bounds(0,num_subregions-1);
	IndexSpace color_is=runtime->create_index_space(ctx,color_bounds);
	IndexPartition ip=runtime->create_equal_partition(ctx,is,color_is);
	runtime->attach_name(ip,"ip");
	LogicalPartition input_lp=runtime->get_logical_partition(ctx,input_lr,ip);
	runtime->attach_name(input_lp,"input_lp");
	ArgumentMap arg_map;
	IndexLauncher init_launcher(INIT_NUM_TASK_ID,color_is,TaskArgument(NULL,0),arg_map);
	init_launcher.add_region_requirement(RegionRequirement(input_lp,0,WRITE_DISCARD,EXCLUSIVE,input_lr));
	init_launcher.region_requirements[0].aff_field(FID_X);
	runtime->execute_index_space(ctx,init_launcher);
	
	runtime->destroy_logical_region(ctx,input_lr);
	runtime->destroy_field_space(ctx,input_fs);
	runtime->destroy_index_space(ctx,is);
}
void init_num_task(const Task * task,const std::vector<PhysicalRegion> & regions,Context ctx,Runtime * runtime)
{
	assert(regions.size() == 1); 
	assert(task->regions.size() == 1);
	assert(task->regions[0].privilege_fields.size() == 1);

  	FieldID fid = *(task->regions[0].privilege_fields.begin());
  	const int point = task->index_point.point_data[0];
  	printf("Initializing field %d for block %d...\n", fid, point);

  	const FieldAccessor<WRITE_DISCARD,double,1> acc(regions[0], fid);
  	// Note here that we get the domain for the subregion for
  	// this task from the runtime which makes it safe for running
  	// both as a single task and as part of an index space of tasks.
  	Rect<1> rect = runtime->get_index_space_domain(ctx,
                  task->regions[0].region.get_index_space());
  	for (PointInRectIterator<1> pir(rect); pir(); pir++)
		acc[*pir] = drand48();
}
int main(int argc,char * * argv)
{
	Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID)
	{
		TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID,"top_level");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		Runtime::preregister_task_variant<top_level_task>(registrar,"top_level");
	}
	{
		TaskVariantRegistrar registrar(INIT_NUM_TASK_ID,"init_num");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		registrar.set_leaf();
		Runtime::preregister_task_variant<init_num_task>(registrar,"init_field");
	}
	{
		TaskVariantRegistrar registrar(SORT_TASK_ID,"simple_sort");
		registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
		registrar.set_leaf();
		Runtime::preregister_task_variant<simple_sort>(registrar,"simple_sort");
	}
	return Runtime::start(argc,argv);
}
