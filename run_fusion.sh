#!/bin/bash
# run_experiments.sh
# 使用例:
#   ./run_experiments.sh MHCG exp --MH_iter 3 --Whole_iter 30 --batch_size 40
#   ./run_experiments.sh all_acceptance myprefix --num_workers 4
#
# 第一引数: モード (MHCG, all_acceptance, distillation) [デフォルト: MHCG]
# 第二引数: 実験名のprefix [デフォルト: exp]
# そのほかの引数はすべて python に渡されます。

# モードの取得（デフォルト: MHCG）
MODE=${1:-MHCG}
# 実験名のprefixの取得（デフォルト: exp）
PREFIX=${2:-exp}
# 先頭の2引数をシフトして、残りは全て python に渡すための引数にする
shift 2

# exp_name を MODE と PREFIX を結合して作成
EXP_NAME="${MODE}_${PREFIX}"

# モードに応じた実行
if [ "$MODE" = "MHCG" ]; then
    echo "=== ${MODE} 実験を開始します (exp_name: ${EXP_NAME}) ==="
    python parallel_communication_field.py --exp_name "$EXP_NAME" "$@"
    echo "=== ${MODE} 実験が終了しました ==="

elif [ "$MODE" = "all_acceptance" ]; then
    echo "=== ${MODE} 実験 (Agent A) を開始します (exp_name: ${EXP_NAME}) ==="
    python parallel_communication_field.py --exp_name "$EXP_NAME" --all_acceptance_agent A "$@"
    echo "=== ${MODE} 実験 (Agent A) が終了しました ==="
    echo ""
    echo "=== ${MODE} 実験 (Agent B) を開始します (exp_name: ${EXP_NAME}) ==="
    python parallel_communication_field.py --exp_name "$EXP_NAME" --all_acceptance_agent B "$@"
    echo "=== ${MODE} 実験 (Agent B) が終了しました ==="

elif [ "$MODE" = "distillation" ]; then
    echo "=== ${MODE} 実験 (Agent A) を開始します (exp_name: ${EXP_NAME}) ==="
    python parallel_communication_field.py --exp_name "$EXP_NAME" --distillation_agent A "$@"
    echo "=== ${MODE} 実験 (Agent A) が終了しました ==="
    echo ""
    echo "=== ${MODE} 実験 (Agent B) を開始します (exp_name: ${EXP_NAME}) ==="
    python parallel_communication_field.py --exp_name "$EXP_NAME" --distillation_agent B "$@"
    echo "=== ${MODE} 実験 (Agent B) が終了しました ==="

else
    echo "Error: 指定されたモード '${MODE}' は無効です。使用可能なモードは MHCG, all_acceptance, distillation です。"
    exit 1
fi
