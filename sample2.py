import pandas as pd
# df = pd.read_csv("DownloadConceptualCaptions/training_imgtxt_cleaned.tsv", sep="\t")

# print(df["caption"].tolist()[:10])
# print(df.shape)

# df = pd.read_csv("DownloadConceptualCaptions/training_imgtxt.tsv", sep="\t")

# print(df["caption"].tolist()[:10])
# print(df.shape)

# exit()

def clean_caption(caption: str) -> str:
    # Remove spaces before periods
    caption = caption.rstrip()
    if caption.endswith(" ."):
        caption = caption[:-2] + "."
    # Add a period if it doesn't have one at the end
    if not caption.endswith("."):
        caption = caption + "."
    return caption

df = pd.read_csv("DownloadConceptualCaptions/validation_imgtxt.tsv", sep="\t")

df["caption"] = df["caption"].apply(clean_caption)

df.to_csv("DownloadConceptualCaptions/validation_imgtxt_cleaned.tsv", sep="\t", index=False)