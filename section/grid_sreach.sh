#!/bin/bash

make clean && make

M=1024
N=1024
K=1024

Mr_Nr=(
    "4 16"
    "5 16"
    "4 20"
    "6 16"
    "4 24"
    "3 16"
)


ops=(0 1 2 3 4 5)

START=32
STEP=16
END=168

echo "开始遍历参数进行测试..."
echo "参数格式: M N K Mc Nc Kc Mr Nr op"

# 日志文件
LOG_FILE="gemm_results.log"
echo "GEMM Benchmark Results" > $LOG_FILE
echo "-----------------------" >> $LOG_FILE

for Mc in $(seq $START $STEP $END); do
    for Nc in $(seq $START $STEP $END); do
        for Kc in $(seq $START $STEP $END); do
            for pair in "${Mr_Nr[@]}"; do
           
                read -r Mr Nr <<< "$pair"
                
                for op in "${ops[@]}"; do
               
                    echo "执行: ./gemm $M $N $K $Mc $Nc $Kc $Mr $Nr $op"
                    
                    echo "Parameters: M=$M N=$N K=$K Mc=$Mc Nc=$Nc Kc=$Kc Mr=$Mr Nr=$Nr op=$op" >> $LOG_FILE
                    
                    ./gemm $M $N $K $Mc $Nc $Kc $Mr $Nr $op >> $LOG_FILE 2>&1
                    
                    echo "-----------------------" >> $LOG_FILE
                    sleep 1
                done
            done
        done
    done
done

echo "测试完成，结果已保存至 $LOG_FILE"
