import sacrebleu

# 類似した参照文を用いたケース
references_similar = [["The quick brown fox jumps over the lazy dog.", "Fast brown foxes leap over lazy dogs."]]
candidate_similar = ["A quick brown fox leaps over lazy dogs."]

# 異なる参照文を用いたケース
references_different = [["The quick brown fox jumps over the lazy dog."], ["Fast brown foxes leap over lazy dogs."]]
candidates_different = ["A quick brown fox leaps over lazy dogs.", "A quick brown fox leaps over lazy dogs."]

# BLEUスコアの計算
bleu_single = sacrebleu.corpus_bleu(candidate_similar, references_similar)
bleu_multi = sacrebleu.corpus_bleu(candidates_different, references_different)

# 結果の出力
print(f"Single Instance BLEU Score: {bleu_single.score}")
print(f"Multiple Instances BLEU Score: {bleu_multi.score}")