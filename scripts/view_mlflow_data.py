import mlflow

mlflow.set_tracking_uri('./mlruns')

# List all experiments
experiments = mlflow.search_experiments()
print("\n📊 All Experiments:")
for exp in experiments:
    print(f"  - {exp.name} (ID: {exp.experiment_id})")
    
    # Get runs for each experiment
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    print(f"    Runs: {len(runs)}")
    
    if len(runs) > 0:
        print(f"    Latest runs:")
        for idx, row in runs.head(3).iterrows():
            print(f"      - Run ID: {row['run_id'][:8]}... | Status: {row['status']}")

# Get all runs from all experiments
all_runs = mlflow.search_runs()
print(f"\n🏃 Total Runs Across All Experiments: {len(all_runs)}")

if len(all_runs) > 0:
    print("\nLatest 5 Runs:")
    print(all_runs[['experiment_id', 'run_id', 'start_time', 'status']].head())

# List registered models
client = mlflow.MlflowClient()
models = client.search_registered_models()
print("\n🤖 Registered Models:")
for model in models:
    print(f"  - {model.name}")
    for version in model.latest_versions:
        print(f"    Version {version.version}: Stage={version.current_stage}")