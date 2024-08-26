import pandas as pd
import sacrebleu
import matplotlib.pyplot as plt

for r in [2]:
    bleu_score_list = []
    bleu_score_std_list = []
    for i in [49]:
        df = pd.read_csv(f'/clipcap-{i:03}.csv')

        refs = df['Reference'].tolist()
        gens = df['Generated'].tolist()

        max_bleu = 0
        max_bleu_idx = 0
        divide = 5
        bleu_list = []
        for j in range(len(refs)//divide):
            ref_for_bleu = [refs[j*divide:j*divide+divide]]
            gens_for_bleu = [gens[j*divide]]
            bleu = sacrebleu.corpus_bleu(gens_for_bleu, ref_for_bleu)
            bleu_list.append(bleu.score)
            max_bleu = max(max_bleu, bleu.score)
            print(ref_for_bleu)
            print(gens_for_bleu)
            print()
            if max_bleu == bleu.score:
                max_bleu_idx = j

        print(max_bleu_idx, max_bleu)
        print(refs[max_bleu_idx*5:max_bleu_idx*5+5])
        print(gens[max_bleu_idx*5:max_bleu_idx*5+5])

        bleu_socre = sum(bleu_list)/len(bleu_list)
        print("Mean BLEU:", bleu_socre)

        bleu_score_list.append(bleu_socre)
        bleu_score_std_list.append(bleu_list)

    plt.clf()

    plt.plot(bleu_score_list)
    plt.xlabel('Epoch')
    plt.ylabel('BLEU')
    plt.savefig(f"cc3m2coco-3/{r}/bleu.png")