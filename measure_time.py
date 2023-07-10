import pandas as pd
import torch
import tqdm
from eval_tools import time_sync
from mivolo.model.create_timm_model import create_model

if __name__ == "__main__":

    face_person_ckpt_path = "/data/dataset/iikrasnova/age_gender/pretrained/checkpoint-377.pth.tar"
    face_person_input_size = [6, 224, 224]

    face_age_ckpt_path = "/data/dataset/iikrasnova/age_gender/pretrained/model_only_age_imdb_4.32.pth.tar"
    face_input_size = [3, 224, 224]

    model_names = ["face_body_model", "face_model"]
    # batch_size = 16
    steps = 1000
    warmup_steps = 10
    device = torch.device("cuda:1")

    df_data = []
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    for ckpt_path, input_size, model_name, num_classes in zip(
        [face_person_ckpt_path, face_age_ckpt_path], [face_person_input_size, face_input_size], model_names, [3, 1]
    ):

        in_chans = input_size[0]
        print(f"Collecting stat for {ckpt_path} ...")
        model = create_model(
            "mivolo_d1_224",
            num_classes=num_classes,
            in_chans=in_chans,
            pretrained=False,
            checkpoint_path=ckpt_path,
            filter_keys=["fds."],
        )
        model = model.to(device)
        model.eval()
        model = model.half()

        time_per_batch = {}
        for batch_size in batch_sizes:
            create_t0 = time_sync()
            for _ in range(steps):
                inputs = torch.randn((batch_size,) + tuple(input_size)).to(device).half()
            create_t1 = time_sync()
            create_taken = create_t1 - create_t0

            with torch.no_grad():
                inputs = torch.randn((batch_size,) + tuple(input_size)).to(device).half()
                for _ in range(warmup_steps):
                    out = model(inputs)

                all_time = 0
                for _ in tqdm.tqdm(range(steps), desc=f"{model_name} batch {batch_size}"):
                    start = time_sync()
                    inputs = torch.randn((batch_size,) + tuple(input_size)).to(device).half()
                    out = model(inputs)
                    out += 1
                    end = time_sync()
                    all_time += end - start

                time_taken = (all_time - create_taken) * 1000 / steps / batch_size
                print(f"Inference {inputs.shape}, steps: {steps}. Mean time taken {time_taken} ms / image")

            time_per_batch[str(batch_size)] = f"{time_taken:.2f}"
        df_data.append(time_per_batch)

    headers = list(map(str, batch_sizes))
    output_df = pd.DataFrame(df_data, columns=headers)
    output_df.index = model_names

    df2_transposed = output_df.T
    out_file = "batch_sizes.csv"
    df2_transposed.to_csv(out_file, sep=",")
    print(f"Saved time stat for {len(df2_transposed)} batches to {out_file}")
