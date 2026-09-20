"""Export existing official WikiText cache to a new three-split local dataset."""
import argparse
from pathlib import Path
from bridge import read,sha_file,save_json,require


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--compare-existing',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from datasets import Dataset,load_from_disk
    info=read(args.cache_dir/'dataset_info.json')
    require(info['config_name']=='wikitext-2-raw-v1','Wrong cached dataset config')
    require(not args.output.exists(),'Use a new output; existing dataset is read-only')
    sources={};datasets={}
    for split in ('train','validation','test'):
        path=args.cache_dir/f'wikitext-{split}.arrow';data=Dataset.from_file(str(path))
        require(data.column_names==['text'] and len(data)==info['splits'][split]['num_examples'],'Cached split metadata differs')
        if split!='test':
            old=load_from_disk(str(args.compare_existing/split));require(old['text']==data['text'],'Cached corpus differs from transferred corpus')
        sources[split]=dict(path=str(path.resolve()),sha256=sha_file(path),rows=len(data));datasets[split]=data
    args.output.mkdir(parents=True)
    for split,data in datasets.items():data.save_to_disk(str(args.output/split))
    save_json(args.output/'provenance.json',dict(source='Existing Salesforce/wikitext wikitext-2-raw-v1 Arrow cache',
        files=sources,train_validation_equal_transferred=True,cache_dataset_info_sha256=sha_file(args.cache_dir/'dataset_info.json')))
    print('DATASET_READY',args.output.resolve(),'No GPU work or network download.')


if __name__=='__main__':main()
