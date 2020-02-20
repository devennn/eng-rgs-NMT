floyd run \
    --gpu --env tensorflow-2.1 \
    --data devennn/datasets/nmt-dataset/5:malay \
    "python main.py eng_malay_p1.txt floydhub"
    # "python main.py eng_malay.txt floydhub"
    # "python main.py eng_rgs.txt floydhub"
