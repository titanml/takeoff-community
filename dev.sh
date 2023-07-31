TAKEOFF_ACCESS_TOKEN=$1
TAKEOFF_MODEL_NAME="meta-llama/Llama-2-13b-chat-hf"
TAKEOFF_DEVICE="cuda"
TAKEOFF_LOG_LEVEL="DEBUG"
CONVERT="true"
RELOAD=1
docker build . -t fab && docker run --gpus all -e TAKEOFF_ACCESS_TOKEN=$TAKEOFF_ACCESS_TOKEN -e TAKEOFF_DEVICE=$TAKEOFF_DEVICE -e CONVERT=$CONVERT -e RELOAD=$RELOAD -e TAKEOFF_MODEL_NAME=$TAKEOFF_MODEL_NAME -e HOST_URL="http://localhost:8000" -v $HOME/.iris_cache:/code/models -v $PWD/app:/code/app -p 8000:80 -it --rm fab
