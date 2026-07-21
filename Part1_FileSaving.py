print("Starting batch execution across all datasets and splits...\n")

for dataset in DATASETS:
    for split in SPLITS:
        print(f"=== Processing Dataset: {dataset} | Split: {split} ===")
        
        try:
            mirna_df, dna_df, raw_train_ids, raw_val_ids, raw_test_ids, missing_dna_ids = load_data(DATA_DIR, dataset, split)

            train_ids = [i for i in raw_train_ids if i in dna_df.index]
            val_ids = [i for i in raw_val_ids if i in dna_df.index]
            test_ids = [i for i in raw_test_ids if i in dna_df.index]

            datasets, scalers = preprocess_pipeline(mirna_df, dna_df, train_ids, val_ids, test_ids, missing_dna_ids)
            X_train, y_train, train_pids = datasets['train']
            X_val, y_val, val_pids = datasets['val']
            X_test, y_test, test_pids = datasets['test']
            X_missing, missing_pids = datasets['missing']

            train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
            val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

            model = MiRNAToDNAPredictor(input_dim=X_train.shape[1], output_dim=y_train.shape[1]).to(DEVICE)
            model = train_model(model, train_loader, val_loader, epochs=100, patience=15)

            model_save_path = os.path.join(MODELS_DIR, f"{dataset}_{split}_mirna_to_dna.pt")
            torch.save(model.state_dict(), model_save_path)

            model.eval()
            with torch.no_grad():
                y_test_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()

            y_test_true = scalers['y'].inverse_transform(y_test)
            y_test_pred = scalers['y'].inverse_transform(y_test_pred_scaled)

            rmse, r2, test_mean_cos_sim, sample_cos_sims = evaluate_predictions(y_test_true, y_test_pred)

            metrics_df = pd.DataFrame([{
                'dataset': dataset,
                'split': split,
                'RMSE': rmse,
                'R2': r2,
                'Mean_Cosine_Similarity': test_mean_cos_sim
            }])
            metrics_path = os.path.join(METRICS_DIR, f"{dataset}_{split}_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)

            if len(X_missing) > 0:
                with torch.no_grad():
                    y_missing_pred_scaled = model(torch.tensor(X_missing, dtype=torch.float32).to(DEVICE)).cpu().numpy()
                y_missing_pred = scalers['y'].inverse_transform(y_missing_pred_scaled)
                imputed_df = pd.DataFrame(y_missing_pred, index=missing_pids, columns=dna_df.columns)
            else:
                imputed_df = pd.DataFrame(columns=dna_df.columns)

            real_dna_ids = list(set(train_ids + val_ids + test_ids))
            real_dna_df = dna_df.loc[real_dna_ids].copy()
            real_dna_df['source'] = 'real'

            if not imputed_df.empty:
                imputed_dna_df = imputed_df.copy()
                imputed_dna_df['source'] = 'imputed'
                full_dna_df = pd.concat([real_dna_df, imputed_dna_df], axis=0)
            else:
                full_dna_df = real_dna_df

            pred_path = os.path.join(PRED_DIR, f"{dataset}_{split}_imputed_DNA.csv")
            full_dna_df.to_csv(pred_path)

            confidence_records = []
            
            for pid in real_dna_ids:
                confidence_records.append({
                    'sample_id': pid,
                    'source': 'real',
                    'cosine_similarity': 1.0,
                    'confidence_score': 1.0
                })

            for pid in missing_pids:
                confidence_records.append({
                    'sample_id': pid,
                    'source': 'imputed',
                    'cosine_similarity': float(test_mean_cos_sim),
                    'confidence_score': max(0.0, float(test_mean_cos_sim))
                })

            conf_df = pd.DataFrame(confidence_records)
            conf_path = os.path.join(CONF_DIR, f"{dataset}_{split}_confidence_scores.csv")
            conf_df.to_csv(conf_path, index=False)

            print(f"Successfully processed and saved outputs for {dataset} (Split {split})")

        except Exception as e:
            print(f"Error processing {dataset} Split {split}: {e}")

