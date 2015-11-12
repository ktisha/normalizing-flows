python -m experiments.vae_mnist -L2 -H500 -B 1500 -E 1000
python -m experiments.vae_mnist -L8 -H500 -B 1500 -E 1000
python -m experiments.vae_mnist -L64 -H500 -B 1500 -E 1000
python -m experiments.vae_mnist -L128 -H500 -B 1500 -E 1000

python -m experiments.nf_mnist -L2 -H500 -B 1500 -E 1000 -F2
python -m experiments.nf_mnist -L2 -H500 -B 1500 -E 1000 -F8
python -m experiments.nf_mnist -L2 -H500 -B 1500 -E 1000 -F32
python -m experiments.nf_mnist -L8 -H500 -B 1500 -E 1000 -F8
python -m experiments.nf_mnist -L8 -H500 -B 1500 -E 1000 -F16
python -m experiments.nf_mnist -L64 -H500 -B 1500 -E 1000 -F8
python -m experiments.nf_mnist -L64 -H500 -B 1500 -E 1000 -F16
python -m experiments.nf_mnist -L128 -H500 -B 1000 -E 1000 -F16
