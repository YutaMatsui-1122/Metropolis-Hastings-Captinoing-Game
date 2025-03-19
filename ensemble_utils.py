from utils import *


# ロジット処理器のリストを作成
rp_processor = RepetitionPenaltyLogitsProcessor(1.7)
nrng_processor = NoRepeatNGramLogitsProcessor(3)
ap_processor = AvoidPeriodLogitsProcessor(period_token_id=tokenizer.encode(".")[0], max_tokens_to_avoid=5)
loss_fct = nn.CrossEntropyLoss(reduction="none")

def ensemble_generate_batch(
    model1,
    model2,
    tokenizer,
    embed1=None,
    embed2=None,
    entry_length=40,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token='.',
    fusion_method="ensemble",  # "ensemble"（等重み） or "packllm_sim"
    tau=1  # temperature for perplexity-based weighting
):
    model1.eval()
    model2.eval()
    stop_token_id = tokenizer.encode(stop_token)[0]
    device = next(model1.parameters()).device

    model1.to(device)
    model2.to(device)

    prefix_length = embed1.shape[1]
    batch_size = embed1.shape[0]
    
    tokens = None
    total_time = 0

    with torch.no_grad():
        generated1 = embed1.to(device)
        generated2 = embed2.to(device)

        for i in range(entry_length):
            start = time.time()
            outputs1 = model1.gpt(inputs_embeds=generated1)
            outputs2 = model2.gpt(inputs_embeds=generated2)

            logits1 = outputs1.logits
            logits2 = outputs2.logits


            # --- 重み付けの条件分岐 ---
            if fusion_method == "ensemble":
                # 等重みアンサンブル
                weights = torch.tensor([[0.5, 0.5]]).repeat(logits1.shape[0], 1).to(device)
            elif fusion_method == "packllm_sim":
                # PackLLMsim：生成済みトークンがない場合は初回は等重み、以降はパープレキシティに基づく重み付け
                if tokens is None :
                    # 初回は等重み
                    weights = torch.tensor([[0.5, 0.5]]).repeat(logits1.shape[0], 1).to(device)
                else:
                    # tokens は (batch, seq_len) なので、最後のトークンを除いて損失を計算
                    loss1 = loss_fct(
                        logits1[:, prefix_length - 1 : -1, :].reshape(-1, logits1.size(-1)),
                        tokens[:, 0:].reshape(-1)
                    ).view(batch_size, -1).sum(dim=1)
                    loss2 = loss_fct(
                        logits2[:, prefix_length - 1 : -1, :].reshape(-1, logits2.size(-1)),
                        tokens[:, 0:].reshape(-1)
                    ).view(batch_size, -1).sum(dim=1)
                    losses = torch.stack([loss1, loss2])
                    weights = F.softmax(-losses / tau, dim=0).T
                # print(weights)
            else:
                # 既定の挙動は等重み
                weights = torch.tensor([[0.5, 0.5]]).repeat(logits1.shape[0], 1).to(device)
            # --- 重み付けここまで ---

            # 各モデルのロジットを重み付けして融合
            logits = weights[:, 0].view(-1, 1, 1) * logits1 + weights[:, 1].view(-1, 1, 1) * logits2
            # print(weights[0],logits[0][0][:5], logits1[0][0][:5], logits2[0][0][:5])

            # 次のトークン予測：全シーケンス中の最後のタイムステップの出力を利用
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            # (後処理: rp_processor, nrng_processor, ap_processor などがあればここで呼び出し)
            if tokens is not None:
                logits = rp_processor(tokens, logits)
                logits = nrng_processor(tokens, logits)
                logits = ap_processor(tokens, logits)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            aux_indices = torch.argsort(sorted_indices, dim=1)
            indices_to_remove = torch.gather(sorted_indices_to_remove, 1, aux_indices)
            logits = torch.where(indices_to_remove, torch.full_like(logits, float('-inf')), logits)    

            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            next_token_embed1 = model1.gpt.transformer.wte(next_token)
            next_token_embed2 = model2.gpt.transformer.wte(next_token)

            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)

            generated1 = torch.cat((generated1, next_token_embed1), dim=1)
            generated2 = torch.cat((generated2, next_token_embed2), dim=1)

            total_time += time.time() - start
            
            # すべてのバッチ行にstop_token_idが存在する場合、ループ終了
            if (tokens == stop_token_id).any(dim=1).all():
                break
            # entry_length - 2回目のループで終了トークンが見つからない場合は補完
            if i == entry_length - 2:
                tokens = torch.cat((tokens, torch.full((tokens.shape[0], 1), stop_token_id, dtype=tokens.dtype).to(device)), dim=1)
                break
        
        output_list = tokens.cpu().numpy().tolist()
        output_text = [tokenizer.decode(output_list[i]) for i in range(len(output_list))]
        output_text = [output_text[i][:output_text[i].find(stop_token)+1] for i in range(len(output_text))]
        output_text = clean_generated_texts(output_text)
    return output_text