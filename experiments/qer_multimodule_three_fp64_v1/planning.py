"""Register modules, shared passes, cache costs and evaluation scope explicitly."""
from bridge import MODULES,METHODS,require


def make_plan(shapes,windows=(128,256),labels=1,length=2048,validation=16,test_windows=1):
    require(labels>0 and length>1 and len(set(windows))==len(windows) and min(windows)>0,'Invalid budget')
    maximum=max(windows);count=len(shapes);require(count>0,'No modules')
    # q/v within a layer share their input. Down-proj is a separate input group.
    inputs={}
    for name,(m,n) in shapes.items():
        group=input_group(name);require(group not in inputs or inputs[group]==n,'Shared input dimension differs');inputs[group]=n
    compact=4*length*maximum*(sum(inputs.values())+labels*sum(m for m,n in shapes.values()))
    dense=8*maximum*labels*sum(m*n for m,n in shapes.values())
    # Persisted statistics, raw factors, one resumable state per iterative
    # method, low-rank factors and deployments. No per-iteration matrix archive.
    factors=len(windows)*sum(8*(10*n*n+8*m*m) for m,n in shapes.values())
    corrections=len(windows)*len(METHODS)*sum(4*m*n+8*64*(m+n) for m,n in shapes.values())
    quantized=8*sum(m*n for m,n in shapes.values())
    return dict(modules=list(shapes),budgets={f'N{n}':[n,labels] for n in windows},methods=METHODS,
        unique_fit_windows=maximum,fit_label_samples=maximum*labels,
        shared_fit_backward_passes=maximum*labels,separate_fit_backward_passes=count*maximum*labels,
        main_eval_backward_passes=0,fit_x_g_GiB=compact/2**30,avoided_fit_S_GiB=dense/2**30,
        estimated_total_persistent_GiB=(compact+factors+corrections+quantized)/2**30,
        storage_note='Approximate tensor payload only; reserve extra space for atomic writes and receipts. Filesystem free space does not imply account quota.',
        reference_teacher_forwards=validation+test_windows,
        validation_single_module_forwards=validation*count*(len(windows)*len(METHODS)+1),
        test_joint_forwards=test_windows*(len(windows)*len(METHODS)+1),
        eval_scope='validation: one target intervention at a time; test: all registered targets jointly; all candidates, no test selection',
        heavy_diagnostics='separate opt-in job; excluded from main workflow')


def input_group(name):
    if name.endswith(('.q_proj','.k_proj','.v_proj')):return name.rsplit('.',1)[0]+'.qkv_input'
    return name+'.input'


def defaults():
    return dict(modules=MODULES,budgets=[128,256],labels_per_window=1,sequence_length=2048,
        methods=METHODS,validation_windows=16,rank=64,offline_workers=2,cache_GiB=2.,
        budget_hours=10.,disk_limit_GiB=512.,minimum_free_GiB=544.,diagnostics=False,
        dataset='wikitext2',calibration_split='train',validation_split='validation',test_split='test')
