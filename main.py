from SeqGAN.train import Trainer
from SeqGAN.get_config import get_config
import time
from insert_error import insert_error
from eva import evaluate
from att_reverse import att_reverse
from rule_sample import rule_sample

config = get_config('config.ini')

path = config["path_pos"]
path_ori = path.strip('_copy')
print(path_ori)
# path_ori = "UIS2"  #Hosp_reverse
# path = "UIS2_copy" #Hosp_reverse
# path_ori = "UIS"  #
# path = "UIS_copy"
f = open('data/save/log_evaluation.txt', 'w')
f.write("")
f.close()
error_rate = 0.1

# reset(path_ori, path)
insert_error(path_ori, path, error_rate)  # On during the first run, off during the reverse run

start_time = time.time()
print("Load the config and define the variables")

flag = 2  # 0 for training SeqGAN, 1 for repairing part, 2 for doing it simultaneously
order = 1  # Order, 1 for positive order, 0 for negative order
att_reverse(path, order)

if flag == 0 or flag == 2:
    trainer = Trainer(order,
                      config["batch_size"],
                      config["max_length"],
                      config["g_e"],
                      config["g_h"],
                      config["d_e"],
                      config["d_h"],
                      config["d_dropout"],
                      config["generate_samples"],
                      path_pos=config["path_pos"],
                      path_neg=config["path_neg"],
                      g_lr=config["g_lr"],
                      d_lr=config["d_lr"],
                      n_sample=config["n_sample"],
                      path_rules=config["path_rules"])
    # Pretraining for adversarial training To run this part against the trained pretrain, batch_size = 32
    print("Start pre-training")
    # insert_error.insert_error(1000)
    trainer.pre_train(g_epochs=config["g_pre_epochs"],  # 50
                      d_epochs=config["d_pre_epochs"],  # 1
                      g_pre_path=config["g_pre_weights_path"],  # data/save/generator_pre.hdf5
                      d_pre_path=config["d_pre_weights_path"],  # data/save/discriminator_pre.hdf5
                      g_lr=config["g_pre_lr"],  # 1e-2
                      d_lr=config["d_pre_lr"])  # 1e-4

    trainer.load_pre_train(config["g_pre_weights_path"], config["d_pre_weights_path"])
    trainer.reflect_pre_train()  # Mapping layer weights to agent

    trainer.train(steps=1,  # A total of 1 big round, each big round in the generator training 1 time, discriminator 
                            # training 1 time
                  g_steps=1,
                  head=10,
                  g_weights_path=config["g_weights_path"],
                  d_weights_path=config["d_weights_path"])

    trainer.save(config["g_weights_path"], config["d_weights_path"])

if flag == 1 or flag == 2:
    trainer = Trainer(order,
                      1,
                      config["max_length"],
                      config["g_e"],
                      config["g_h"],
                      config["d_e"],
                      config["d_h"],
                      config["d_dropout"],
                      config["generate_samples"],
                      path_pos=config["path_pos"],
                      path_neg=config["path_neg"],
                      g_lr=config["g_lr"],
                      d_lr=config["d_lr"],
                      n_sample=config["n_sample"],
                      path_rules=config["path_rules"])
    trainer.load(config["g_weights_path"], config["d_weights_path"])

    rule_len = rule_sample(config["path_rules"], path, order)
    # trainer.generate_rules(config["path_rules"], config["generate_samples"])  
    # Production of rule sequence, i.e. rules.txt

    trainer.train_rules(rule_len, config["path_rules"])  # For production rules, generate rules_final.txt from rules.txt

    trainer.filter(config["path_pos"])

    f = open('data/save/log.txt', 'w')
    f.write("")
    f.close()
    att_reverse(path, 1)
    trainer.repair(config["path_pos"])  # Used for two-way repair, the rules are from rules_final.txt, the repair data 
    # is Hosp_rules_copy, which is a full backup of Hosp_rules + random noise
    # trainer.repair_SeqGAN()

end_time = time.time()
dtime = end_time - start_time
print("The runtime of this fix is", dtime, "Error rate is", error_rate)

evaluate(path_ori, path)
# error_rate = error_rate + 10
# while (error_rate < 0.9):
#
# trainer.test()
#
# trainer.generate_txt(config["g_test_path"], config["generate_samples"])
# trainer.predict_rules() #Already useless, only as a test