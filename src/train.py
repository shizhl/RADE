import os.path
import sys

import torch

sys.path.append('../')
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
from transformers import AdamW,get_linear_schedule_with_warmup
from RefTrainer.modeling import *
from RefTrainer.utils import *
from RefTrainer.RefDataset import *
from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICE']='0'
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("-project_name",  default='Evaluation', type=str)
    parser.add_argument("-model_path", type=str)

    parser.add_argument("-ref_enc_path", type=str)
    parser.add_argument("-data_path", type=str )
    parser.add_argument("-output_dir",type=str )
    parser.add_argument("-ckpt",type=str )
    parser.add_argument("-stage",
                        default='2',
                        choices=['1', '2'],
                        type=str,
                        )
    parser.add_argument("-train_batchsize", default=4,type=int)
    parser.add_argument("-accumulate_step", default=32,type=int)
    parser.add_argument("-epoch", default=10000 ,type=int)
    parser.add_argument("-lr", default=2e-5,type=float)
    parser.add_argument("-adv",  default=True, type=str2bool)

    parser.add_argument("-device", default='3',type=str)
    parser.add_argument("-eval_time_per_epoch", default=1,  type=float)
    parser.add_argument("-output", default='./output/')
    parser.add_argument("-input_len", default=512,type=int)
    parser.add_argument("-output_len", default=256,type=int)
    parser.add_argument("-test_batchsize", default=8,type=int)
    parser.add_argument("-num_warmup_steps", default=0.01,type=float)

    parser.add_argument("-interaction",  default=True, type=str2bool)
    parser.add_argument("-upload",  default=False, type=str2bool)
    parser.add_argument("-is_store",  default=True, type=str2bool)
    parser.add_argument("-train",  default=True, type=str2bool)
    parser.add_argument("-joint",  default=True, type=str2bool)
    parser.add_argument("-test",  default=True, type=str2bool)

    config=parser.parse_args()
    # config=baseConfig(config)
    print(config)

    if config.stage=='1':
        print("train simulator from scratch...")
        model_path=config.model_path

    else:
        print("finetune simulator on downstream task...")
        model_path=config.ckpt
        print(f"load from {model_path}...")


    if 'bart' in model_path:
        model = MySimulator(model_path=model_path,
                              input_len=config.input_len,
                              output_len=config.output_len,
                              device=config.device,
                              model_type='Bart')

    elif 'T5' in model_path:
        model = MySimulator(model_path=model_path,
                              input_len=config.input_len,
                              output_len=config.output_len,
                              device=config.device,
                              model_type='T5')

    elif 'Pro' in model_path:
        model = MySimulator(model_path=model_path,
                            input_len=config.input_len,
                            output_len=config.output_len,
                            device=config.device,
                            model_type='prophetnet')

    elif 'gpt' in model_path.lower():
        model = GPT2Simulator(model_path=model_path,
                            input_len=config.input_len,
                            output_len=config.output_len,
                            device=config.device,
                            model_type='gpt')

    model.to(f"cuda:{config.device}")

    optimizer = AdamW(model.parameters(), lr=config.lr)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    lim=2000000
    train_dataset = RefDataset(load_data(os.path.join(config.data_path,'train.jsonl'))[:lim], tokenizer, config)
    test_dataset = RefDataset(load_data(os.path.join(config.data_path,'test.jsonl'))[:lim], tokenizer, config)
    valid_dataset = RefDataset(load_data(os.path.join(config.data_path,'test.jsonl'))[:lim], tokenizer, config)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.train_batchsize,
                                  collate_fn=RefDataset.collect_fn_data,drop_last=True,shuffle=True,num_workers=0)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size= config.test_batchsize, collate_fn=RefDataset.collect_fn_data,drop_last=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size= config.test_batchsize, collate_fn=RefDataset.collect_fn_data,drop_last=True)

    all_step=config.epoch*len(train_dataloader)//(config.train_batchsize*config.accumulate_step)

    lr_scheduler=get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=int(all_step*config.num_warmup_steps),
                                                 num_training_steps=all_step)
    cnt = 0
    score=0
    d=[]
    accumulate_step = config.accumulate_step
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config.epoch):
        torch.cuda.empty_cache()
        tk0 = tqdm(train_dataloader)
        losses = []
        model.train()
        for batch in (tk0):
            cnt+=1
            with torch.cuda.amp.autocast():
                if config.joint==False:
                    batch.pop('labels')
                elif epoch<1 and cnt<len(train_dataloader):
                    batch.pop('labels')
                output = model(batch)
                loss = output['loss']
                if loss.isnan():
                    continue
                losses.append(loss.item())  #
                loss /= accumulate_step
                # loss.backward()
            scaler.scale(loss).backward()

            # if self.adv:
            #     self.fgm.attack()  #
            #     loss_adv = self.model(batch)['loss']
            #     loss_adv.backward()  #
            #     self.fgm.restore()  #

            if cnt % accumulate_step == 0:
                # self.optimizer.step()
                # self.lr_scheduler.step()
                # self.optimizer.zero_grad()
                scaler.step(optimizer)
                scaler.update()
                # self.optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            tk0.set_postfix(loss=losses[-1], avg_loss=format(sum(losses) / len(losses), '.4f'))

        # model.model.save_pretrained(os.path.join(config.output_dir,f'epoch{epoch}'))
        def eval(data_loader):
            preds_resp, labels_resp = [], []
            preds_score, labels_score = [], []
            ctx = []
            model.eval()
            with torch.no_grad():
                for batch in tqdm(data_loader):
                    response, score = model.generate(batch)
                    response = tokenizer.batch_decode(response, skip_special_tokens=True)
                    if type(batch['labels']) != list:
                        labels_score.extend(batch['labels'].tolist())
                    else:
                        labels_score.extend(batch['labels'])
                    preds_score.extend(score)
                    preds_resp.extend(response)
                    labels_resp.extend(batch['reference'])
                    ctx.extend(batch['context'])
            labels_score = [line[0] for line in labels_score]
            preds_resp = [line if line != '' else 'no word' for line in preds_resp]
            pearson = scipy.stats.pearsonr(preds_score, labels_score)[0]
            spearmanr = scipy.stats.spearmanr(preds_score, labels_score)[0]
            kendalltau = scipy.stats.kendalltau(preds_score, labels_score)[0]
            result = {"pearson": pearson, "spearmanr": spearmanr, 'kendalltau': kendalltau,
                      'score': (pearson + spearmanr + kendalltau) / 3}
            # try:
            #     result.update(
            #         eval_specific(preds_resp, labels_resp, specific=['eval_rouge', 'eval_bleu', ],
            #                       device=config.device))
            # except:
            #     print("error")
            # filename=os.path.join(config.output_dir,f'epoch{epoch}',f'result.txt')
            # with open(filename, 'w') as f:
            #     f.write(json.dumps(result, indent=4) + '\n')
            #     for line1, line2 in zip(preds_resp, labels_resp):
            #         f.write(line1 + ' [SPE] ' + line2 + '\n')
            # print(f'write result to {filename}...')
            print(result)
            return result,preds_score
        eval(test_dataloader)
