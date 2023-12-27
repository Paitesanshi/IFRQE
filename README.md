# Would You Like Your Data to Be Trained? A User Controllable Recommendation Framework

A PyTorch implementation of Would You Like Your Data to Be Trained? A User Controllable Recommendation Framework (IFRQE) in AAAI 2024.

[project page](https://paitesanshi.github.io/IFRQE/)



# Illustration

An illustration of our recommendation paradigm IFRQE, where the users can explicitly indicate their disclosing willingness, and the model needs to trade-off the recommendation quality and user willingness. IFRQE first finds the best strategy that balances the recommendation quality and user willingness. Then the user interactions are disclosed according to the optimal strategy. At last, the recommender model is trained based on the disclosed interactions.



# Usage

1.Install the required packages according to requirements.txt.

```bash
pip install -r requirements.txt
```


2.Prepare the datasets.

(1) Directly download the processed datasets used in this paper:

[Steam](https://drive.google.com/drive/folders/1aBQZdY8337420WM4nOB7uc_rzVnM4pS2?usp=sharing)
[Diginetica](https://drive.google.com/drive/folders/1GbOBO0UsVGD4CDo3KnL-Y7EHhuqUs7Q8?usp=sharing)
[Amazon](https://drive.google.com/drive/folders/1MnpoXYa-EAOJ4d3xK-khJtMaUtCr1Op0?usp=sharing)

(2) Use your own datasets:

Ensure that your data is organized according to the format: `user_id:token, item_id:token, timestamp:float`.


3.Rename the dataset by `dataset-name.inter`, and put it into the folder `./dataset/dataset-name/` (note: replace "dataset-name" with your own dataset name).


4.Run main.py to train our model, where the training parameters can be indicated through the config file.

For example:

```bash
python main.py --model=MF --dataset=diginetica --config_files=mf_diginetica.yaml
```

The parameter tuning ranges in our paper are as follows:

| Parameter              | Range                  |
| ---------------------- | ---------------------- |
| learning rate          | [0.001,0.01,0.05]      |
| batch size             | [1024,2048,4096]       |
| embedding size         | [64,128,256]           |
| drop ratio             | [0.01,0.1,0.2]         |
| lambda                 | [0.1,0.5,1]            |
| iteration number M     | [1,3,5,10]             |
| training epochs        | [50,100,150]           |
| L                      | [500,1000,2000]        |
| T                      | [1,2,3,4,5,6,7,8,9,10] |
| Nj for computing H^-1: | [10，20，30]           |




# Detailed file structure

`IFRQE`:  the main program that contains our model.

| catalog             | Description                                   |
| ------------------- | --------------------------------------------- |
| `IFRQE.config`      | configed program                              |
| `IFRQE.data`        | dataloading and handling program              |
| `IFRQE.evaluator`   | evaluation program                            |
| `IFRQE.model`       | our methods,sampler model and anchor model    |
| `IFRQE.properties`  | predefined configuration of model and dataset |
| `IFRQE.quick_start` | quick start program                           |
| `IFRQE.sampler`     | data sampler                                  |
| `IFRQE.trainer`     | training,validating and testing program       |
| `IFRQE.utils`       | utils used to construct model                 |

| catalog   | Description         |
| --------- | ------------------- |
| `dataset` | predefined dataset  |
| `log`     | running information |
| `saved`   | saved model pth     |
| `asset`   | model asset         |
