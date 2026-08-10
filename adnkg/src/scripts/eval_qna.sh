DATAPATH="../dataset/LiHua-World/LiHua-World"
WORKDIR="../output/LiW-minirag-9b"
QUERYPATH="../dataset/LiHua-World/qa/query_set.csv"

# python3 "./reproduce/Step_1_QA.py" \
#     --outputpath $WORKDIR/output_QA_raw.csv \
#     --llm-host http://172.16.125.141:11434 \
#     --llm-model qwen3.5:9b \
#     --workingdir $WORKDIR \
#     --datapath $DATAPATH \
#     --querypath $QUERYPATH --graph-storage NetworkXStorage

#################LiWorl#########################
# python3 ./reproduce/Step_X2_QnA_Liworld.py \
#     --llm-host http://localhost:11434 \
#     --llm-model qwen3.5:9b \
#     --workingdir ../output/LiW-minirag-9b \
#     --querypath ../dataset/LiHua-World/qa/query_set.csv \
#     --outputpath ../output/LiW-minirag-9b/output_QA_raw.csv \
#     --graph-storage NetworkXStorage \
#     --mode simrag

# python3 ./reproduce/Step_X2_QnA_Liworld.py \
#     --llm-host http://localhost:11434 \
#     --llm-model qwen3.5:9b \
#     --workingdir ../output/LiW-fl-9b \
#     --querypath ../dataset/LiHua-World/qa/query_set.csv \
#     --outputpath ../output/LiW-fl-9b/output_QA_raw.csv \
#     --graph-storage NetworkXStorage \
#     --mode simrag

#################TranHist#########################
# python3 ./reproduce/Step_X2_QnA.py \
#     --llm-host http://172.16.125.141:11434 \
#     --llm-model qwen3.5:35b \
#     --workingdir ../output/VNH3045-fl-35b-r2 \
#     --querypath ../dataset-v1/.fined/query_set_VNHXX3045.csv \
#     --outputpath ../output/VNH3045-fl-35b-r2/output_QA_raw.csv \
#     --graph-storage NetworkXStorage \
#     --mode simrag

# python3 ./reproduce/Step_X2_QnA.py \
#     --llm-host http://172.16.125.141:11434 \
#     --llm-model qwen3.5:35b \
#     --workingdir ../output/VNH3045-minirag-35b \
#     --querypath ../dataset-v1/.fined/query_set_VNHXX3045.csv \
#     --outputpath ../output/VNH3045-minirag-35b/output_QA_raw.csv \
#     --graph-storage NetworkXStorage \
#     --mode simrag

# python3 ./reproduce/Step_X2_QnA.py \
#     --llm-host http://172.16.125.141:11434 \
#     --llm-model qwen3.5:9b \
#     --workingdir ../output/VNH3045-onlyschema-9b \
#     --querypath ../dataset-v1/.fined/query_set_VNHXX3045.csv \
#     --outputpath ../output/VNH3045-onlyschema-9b/output_QA_raw.csv \
#     --graph-storage NetworkXStorage \
#     --mode simrag

python3 ./reproduce/Step_X2_QnA.py \
    --llm-host http://172.16.125.141:11434 \
    --llm-model qwen3.5:9b \
    --workingdir ../output/VNH3045-fl-noschema-9b \
    --querypath ../dataset-v1/.fined/query_set_VNHXX3045.csv \
    --outputpath ../output/VNH3045-fl-noschema-9b/output_QA_raw.csv \
    --graph-storage NetworkXStorage \
    --mode simrag
