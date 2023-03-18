from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import ByteLevel, CharDelimiterSplit
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from pathlib import Path


PATH_TO_FASTA = Path('data/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna')
VOCAB_SIZE = 32000


def read_fasta(path: Path):
    genome = ''
    with open(path) as file:
        for line in file.readlines():
            if not line.startswith('>'):
                genome += line.strip('\n ')
    return genome


def tokenize(genome: str):
    tokenizer = Tokenizer(BPE())
    # tokenizer.normalizer = Lowercase
    # tokenizer.pre_tokenizer = CharDelimiterSplit(' ')
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, show_progress=False)
    tokenizer.train_from_iterator(genome, trainer)

    return tokenizer


if __name__ == '__main__':
    genome = read_fasta(PATH_TO_FASTA)
    tokenizer = tokenize([genome])
    # encoded = tokenizer.encode('DDDDAAAADDDDBBAABBAUADBBAOIJLKJMLKJHHABABBABBBABBBABBABBABABBA')
    tokenizer.save('tokenizer_mus_BPE')         #This will save the tokenizer.json and merges.txt files 
    tokenizer.save_pretrained('pretrained')
    tokenizer.save_model("genome_tokenizer")    
    # print(tokenizer.get_vocab())
