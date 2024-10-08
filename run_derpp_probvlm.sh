#!/bin/bash

# cross_modal_lambdaの値を0.5から1.5まで、0.1ずつ変えて実行
for cml in $(seq 0.5 0.1 1.5); do
    # トレーニング: basemodelがcc3mの場合
    python derpp_probvlm.py --te_update_epochs 10 --num_workers 8 --device cuda:3 --exp_name objective_test_derpp --mode DERPP --basemodel cc3m --cross_modal_lambda $cml || true
    
    # トレーニング: basemodelがcocoの場合
    python derpp_probvlm.py --te_update_epochs 10 --num_workers 8 --device cuda:3 --exp_name objective_test_derpp --mode DERPP --basemodel coco --cross_modal_lambda $cml || true

    # テスト: cc3m2cocoのテスト実行
    python test_derpp_probvlm.py --num_workers 8 --exp_name objective_test_derpp_cc3m2coco_cml_$cml || true
    
    # テスト: coco2cc3mのテスト実行
    python test_derpp_probvlm.py --num_workers 8 --exp_name objective_test_derpp_coco2cc3m_cml_$cml || true
done
