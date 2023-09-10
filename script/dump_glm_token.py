from transformers import AutoTokenizer
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-d',
                    '--dir',
                    type=str,
                    help="model dir, including model and tokenizer")
parser.add_argument('-o',
                    '--output',
                    type=str,
                    default='glm_vocab',
                    help="output file name")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.dir, trust_remote_code=True)
vocab = tokenizer.get_vocab()
fp = open(args.output, 'w', encoding='utf-8')
json.dump(vocab, fp, indent=2, ensure_ascii=False)

#model = AutoModel.from_pretrained(args.dir, trust_remote_code=True, device='cuda')
#model = model.eval()

#response, history = model.chat(tokenizer, "你好", history=[])
#print(response)
#response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
#print(response)
