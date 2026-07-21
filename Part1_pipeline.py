def extract_sample_ids(file_path):
    df = pd.read_csv(file_path)
    for col in ['sample_id', 'sample', 'id', 'ID', 'Sample']:
        if col in df.columns:
            return df[col].dropna().astype(str).tolist()
    return df.iloc[:, 0].dropna().astype(str).tolist()

def load_data(data_dir, dataset, split):
    split_folder = os.path.join(data_dir, dataset, str(split))
    dataset_folder = os.path.join(data_dir, dataset)
    
    mirna_path = os.path.join(split_folder, 'miRNA.csv')
    if not os.path.exists(mirna_path):
        mirna_path = os.path.join(dataset_folder, 'miRNA.csv')
        
    dna_path = os.path.join(split_folder, 'DNA.csv')
    if not os.path.exists(dna_path):
        dna_path = os.path.join(dataset_folder, 'DNA.csv')

    if not os.path.exists(mirna_path) or not os.path.exists(dna_path):
        raise FileNotFoundError(f"Could not locate miRNA.csv or DNA.csv in {split_folder} or {dataset_folder}")
        
    mirna_df = pd.read_csv(mirna_path, index_col=0)
    dna_df = pd.read_csv(dna_path, index_col=0)

    mirna_df.index = mirna_df.index.astype(str)
    dna_df.index = dna_df.index.astype(str)

    train_label_path = os.path.join(split_folder, 'label_train_paired.csv')
    val_label_path = os.path.join(split_folder, 'label_val_paired.csv')
    test_label_path = os.path.join(split_folder, 'label_test_paired.csv')

    if not os.path.exists(train_label_path):
        train_label_path = os.path.join(split_folder, f'label_train_paired_{split}.csv')
    if not os.path.exists(val_label_path):
        val_label_path = os.path.join(split_folder, f'label_val_paired_{split}.csv')
    if not os.path.exists(test_label_path):
        test_label_path = os.path.join(split_folder, f'label_test_paired_{split}.csv')

    train_ids = extract_sample_ids(train_label_path)
    val_ids = extract_sample_ids(val_label_path)
    test_ids = extract_sample_ids(test_label_path)

    missing_dna_ids = sorted(list(set(mirna_df.index) - set(dna_df.index)))

    return mirna_df, dna_df, train_ids, val_ids, test_ids, missing_dna_ids

def preprocess_pipeline(mirna_df, dna_df, train_ids, val_ids, test_ids, missing_dna_ids):
    X_train_raw = mirna_df.loc[train_ids]
    y_train_raw = dna_df.loc[train_ids]
    
    X_val_raw = mirna_df.loc[val_ids]
    y_val_raw = dna_df.loc[val_ids]
    
    X_test_raw = mirna_df.loc[test_ids]
    y_test_raw = dna_df.loc[test_ids]
    
    X_missing_raw = mirna_df.loc[missing_dna_ids] if len(missing_dna_ids) > 0 else pd.DataFrame()

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_x.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw)

    X_train = np.nan_to_num(X_train)
    y_train = np.nan_to_num(y_train)

    X_val = np.nan_to_num(scaler_x.transform(X_val_raw))
    y_val = np.nan_to_num(scaler_y.transform(y_val_raw))

    X_test = np.nan_to_num(scaler_x.transform(X_test_raw))
    y_test = np.nan_to_num(scaler_y.transform(y_test_raw))

    X_missing = np.nan_to_num(scaler_x.transform(X_missing_raw)) if not X_missing_raw.empty else np.array([])

    scalers = {'x': scaler_x, 'y': scaler_y}
    datasets = {
        'train': (X_train, y_train, train_ids),
        'val': (X_val, y_val, val_ids),
        'test': (X_test, y_test, test_ids),
        'missing': (X_missing, missing_dna_ids)
    }
    
    return datasets, scalers

class MiRNAToDNAPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MiRNAToDNAPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=15):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                pred = model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item() * bx.size(0)

        val_loss /= len(val_loader.dataset)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_weights)
    return model

def evaluate_predictions(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    cos_sims = np.array([
        cosine_similarity(y_true[i].reshape(1, -1), y_pred[i].reshape(1, -1))[0, 0]
        for i in range(len(y_true))
    ])
    mean_cos_sim = np.mean(cos_sims)
    
    return rmse, r2, mean_cos_sim, cos_sims
