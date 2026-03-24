# 遠端測試指令
因為用遠端測試，關掉視窗就消失了，所以要用nohup

## 1. Pretrain

```bash
nohup python -u pretrain.py --dataset online-boutique --surr-epochs 100 --expl-epochs 150 > pretrain_ob.log 2>&1 &
```
跑全部 pretrain 腳本
```bash
nohup bash retrain_all.sh > retrain_all.log 2>&1 &
```

## 2. Finetune

```bash
nohup python -u finetune.py --dataset online-boutique --surr-epochs 100 --expl-epochs 150 > finetune_ob.log 2>&1 &
```

## 3. Evaluate

```bash
nohup python -u main.py --dataset online-boutique --surr-epochs 100 --expl-epochs 150 > main_ob.log 2>&1 &
```
指令解釋與替換
    dataset 資料集選擇 : online-boutique, RE1-ob, RE2-ob, RE3-ob
    surr-epochs 訓練輪數 : 100
    expl-epochs 訓練輪數 : 150
    > log檔名稱 : pretrain_ob.log, finetune_ob.log, main_ob.log
    & : 背景執行
    -u : unbuffered output
    2>&1 : standard error to standard output

## 4. 檢查 log

```bash
tail -f pretrain_ob.log
tail -f finetune_ob.log
tail -f main_ob.log
tail -f retrain_all.log
```

```bash
cat pretrain_ob.log
cat retrain_all.log
```
指令解釋與替換
    tail -f : 顯示最後幾行，並持續追蹤更新
    cat : 顯示全部內容

## 5. 停止背景程式

```bash
ps aux | grep pretrain.py
ps aux | grep finetune.py
ps aux | grep main.py
```

## 6. 停止背景程式

```bash
kill -9 <PID>
```     

## 7. 檢查 GPU 使用狀況

```bash
nvidia-smi
```

## 8. 檢查 CPU 使用狀況

```bash
top
```

## 9. 檢查在執行的程式

```bash
ps aux | grep pretrain.py
ps aux | grep finetune.py
ps aux | grep main.py
```
