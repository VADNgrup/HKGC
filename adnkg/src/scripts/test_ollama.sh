curl http://172.16.125.200:33033/api/chat -d '{
  "model": "qwen3.5:4b",
  "messages": [
    {
      "role": "user",
      "content": "Viết một bài luận 2000 từ về lịch sử Nhật Bản"
    }
  ],
  "think": false,
  "stream": false
}'
