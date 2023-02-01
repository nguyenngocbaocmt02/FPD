import umap

def read_instance(save_path):
    dp = mk.DataPanel.read(os.path.join(save_path, "data"))
    with open(os.path.join(save_path, "config.yaml"), "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=SafeLoader)
    return dp, cfg   

def draw_instance(instance_folder_path, image_save_path):
    # UMAP
    dp, cfg = read_instance(instance_folder_path)
    emb = dp["emb"]
    pattern = dp["pattern"]

    in_pattern_samples = []
    in_pattern_pattern = []
    for i, e in enumerate(emb):
        if pattern[i] != -1:
            in_pattern_samples.append(emb[i])
            in_pattern_pattern.append(pattern[i])
    in_pattern_samples = np.array(in_pattern_samples)
    in_pattern_pattern = np.array(in_pattern_pattern)
    
    embedding = umap.UMAP().fit_transform(sentences["bert_v"].tolist())
    sentences=sentences.assign(umap_x=embedding[:,0])
    sentences=sentences.assign(umap_y=embedding[:,1])
    for c in sentences['category'].unique():
    plt.scatter(sentences[sentences.category == c]["umap_x"], sentences[sentences.category == c]["umap_y"], label=c)
    plt.legend()
    plt.title("UMAP")
    plt.savefig('umap.png')
        