import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import copy
from lib.utils.federated_utils import *
from lib.utils.avgmeter import AverageMeter
from train.utils import *
from train.loss import *
from train.context import disable_tracking_bn_stats
from train.ramps import exp_rampup

def train_irc(train_dloader_list, test_dloader_list, model_list, classifier_list, optimizer_list, classifier_optimizer_list, epoch, writer,
        num_classes, domain_weight, source_domains, batchnorm_mmd, batch_per_epoch, confidence_gate_begin,
        confidence_gate_end, communication_rounds, total_epochs, malicious_domain, attack_level, args=None, pre_models=None, pre_classifiers=None, mean=None):
    task_criterion = nn.CrossEntropyLoss().cuda()
    logsoftmax = nn.LogSoftmax(dim=1).cuda()
    cos = nn.CosineSimilarity(dim=1).cuda()
    tau = 0.05
    source_domain_num = len(train_dloader_list[1:])
    for model in model_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()
    # If communication rounds <1,
    # then we perform parameter aggregation after (1/communication_rounds) epochs
    # If communication rounds >=1:
    # then we extend the training epochs and use fewer samples in each epoch.
    if communication_rounds in [0.2, 0.5]:
        model_aggregation_frequency = round(1 / communication_rounds)
    else:
        model_aggregation_frequency = 1
    
    # # train local source domain models
    for f in range(model_aggregation_frequency):
        current_domain_index = 0
        # Train model locally on source domains
        for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list[1:],
                                                                                    model_list[1:],
                                                                                    classifier_list[1:],
                                                                                    optimizer_list[1:],
                                                                                    classifier_optimizer_list[1:]):
            
            # check if the source domain is the malicious domain with poisoning attack
            source_domain = source_domains[current_domain_index]
            current_domain_index += 1
            if source_domain == malicious_domain and attack_level > 0:
                poisoning_attack = True
            else:
                poisoning_attack = False
            
            for i, (image_s, label_s) in enumerate(train_dloader):
                if i >= batch_per_epoch:
                    break
                image_s_w = image_s[0].cuda()
                image_s_s = image_s[1].cuda()
                label_s = label_s.long().cuda()
                true_label = label_s
                if poisoning_attack:
                    # perform poison attack on source domain
                    corrupted_num = round(label_s.size(0) * attack_level)
                    # provide fake labels for those corrupted data
                    label_s[:corrupted_num, ...] = (label_s[:corrupted_num, ...] + 1) % num_classes
                # reset grad
                optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                # each source domain do optimize

                feature_s, _ = model(image_s_w)
                if args.pj == 1:
                    _, feature_cur = model(image_s_s)
                else:
                    feature_cur, _ = model(image_s_s)
                output_s = classifier(feature_s)
                
                # label smooth
                log_probs = torch.log_softmax(output_s, dim=1)
                label_s = torch.zeros(log_probs.size()).scatter_(1, label_s.unsqueeze(1).cpu(), 1)
                label_s = label_s.cuda()
                alpha = 0.1
                label_s = (1-alpha) * label_s + alpha / num_classes
                task_loss_s = (- label_s * log_probs).mean(0).sum()

                # Instance-Instance similarity graph
                feature_sims = []
                sort_idxs = []
                target_sim = []
                target_gcc = []
                with torch.no_grad():
                    # calculate the instance2instance similarity, from offline local target domain model (weak augmentation)
                    temp = F.normalize(pre_models[0](image_s_w)[0])
                    sim = torch.matmul(temp, torch.t(temp))
                    idx = torch.sort(-sim)[1]
                    feature_sims.append(sim)
                    sort_idxs.append(idx)
                    k=12
                    target_tmp = torch.zeros_like(sim)
                    for j in range(image_s_w.size(0)):
                        target_tmp[j][idx[j][:k]] = 1.0
                    target_sim.append(target_tmp)

                    for i in range(len(feature_sims)):
                        if i == 0:
                            target_gcc = feature_sims[i]
                        else:
                            target_gcc += feature_sims[i]
                    
                    tau = args.s_tau
                    target_gcc = target_gcc/tau
                    target_gcc = F.softmax(target_gcc, dim=1)

                    # calculate the instance2instance similarity, from online local source domain model (weak augmentation)
                    intra_temp = F.normalize(model(image_s_w)[0])
                    intra_sim = torch.matmul(intra_temp, torch.t(intra_temp))
                    intra_gcc = intra_sim/tau
                    intra_gcc = F.softmax(intra_gcc, dim=1)

                # calculate the instance2instance similarity, from online local source domain model (weak augmentation)
                online_temp = F.normalize(model(image_s_w)[0])
                online_sim = torch.matmul(online_temp, torch.t(online_temp))
                online_sim_tmp = logsoftmax(online_sim)

                # inter-irc loss, online local source model (weak augmentation) -> offline local target model (weak augmentation), cross domains
                inter_gcc_loss = online_sim_tmp * target_gcc
                inter_gcc_loss = - inter_gcc_loss.mean(0).sum()
                
                # calculate the instance2instance similarity, from online local source domain model (strong augmentation)
                online_intra_temp = F.normalize(model(image_s_s)[0])
                online_intra_sim = torch.matmul(online_intra_temp, torch.t(online_intra_temp))
                online_intra_sim_tmp = logsoftmax(online_intra_sim)

                # intra-irc loss, online local source model (weak augmentation) -> online local source model (strong augmentation), cross data views
                intra_gcc_loss = online_intra_sim_tmp * intra_gcc
                intra_gcc_loss = - intra_gcc_loss.mean(0).sum()

                loss = task_loss_s + args.s_inter * inter_gcc_loss + args.s_intra * intra_gcc_loss
                loss.backward()
                optimizer.step()
                classifier_optimizer.step()

    # the epochs of local target domain
    epochs_target = 1
    # 模型聚合权重
    target_weight = [0, 0]
    consensus_focus_dict = {}
    
    # train local target domain model
    for f in range(epochs_target):
        # train local target domain model by pseudo-labeling strategy, such as knowledge vote strategy (KD3A)
        confidence_gate = (confidence_gate_end - confidence_gate_begin) * (epoch / total_epochs) + confidence_gate_begin
        
        for i in range(1, len(train_dloader_list)):
            consensus_focus_dict[i] = 0
        
        for i, (image_t, label_t) in enumerate(train_dloader_list[0]):
            if i >= batch_per_epoch:
                break
            optimizer_list[0].zero_grad()
            classifier_optimizer_list[0].zero_grad()

            image_w = image_t[0].cuda()
            image_s = image_t[1].cuda()
            
            # knowledge vote
            with torch.no_grad():
                knowledge_list = [torch.softmax(classifier_list[i](model_list[i](image_w)[0]), dim=1).unsqueeze(1) for
                                i in range(0, len(classifier_list))]
                knowledge_list = torch.cat(knowledge_list, 1)
            _, kv_pred, kv_mask = knowledge_vote(knowledge_list, confidence_gate,
                                                                num_classes=num_classes)
            target_weight[0] += torch.sum(kv_mask).item()
            target_weight[1] += kv_mask.size(0)
            consensus_focus_dict = calculate_consensus_focus(consensus_focus_dict, knowledge_list, confidence_gate,
                                                        source_domain_num, num_classes)
            
            graph_weights = []
            sum_weights = 0.0
            for k,v in consensus_focus_dict.items():
                graph_weights.append(v)
                sum_weights += v
            for i in range(len(graph_weights)):
                graph_weights[i]/=(sum_weights + 1e-5)

            # choose pseudo-labeling strategies
            pseudo_label = []
            label_mask = []

            # if args.pl == 1:
            #     # Max predictioin
            #     pseudo_label = max_pred
            #     label_mask = max_pred_mask
            # elif args.pl == 2:
            #     # mean prediction
            #     pseudo_label = mean_pred
            #     label_mask = mean_pred_mask
            # elif args.pl == 3:
            #     # knowledge vote
            #     pseudo_label = kv_pred
            #     label_mask = kv_mask
            # else:
            #     pseudo_label = weighted_mean_pred
            #     label_mask = weighted_mean_pred_mask

            # knowledge vote, from KD3A
            pseudo_label = kv_pred
            label_mask = kv_mask

            # # Mixup strategy
            # lam = np.random.beta(2, 2)
            # batch_size = image_w.size(0)
            # index = torch.randperm(batch_size).cuda()
            # mixed_image = lam * image_w + (1 - lam) * image_w[index, :]
            # mixed_label = lam * pseudo_label + (1 - lam) * pseudo_label[index, :]
            # feature_t, _ = model_list[0](mixed_image)
            # output_t_cls = classifier_list[0](feature_t)
            # output_t = torch.log_softmax(output_t_cls, dim=1)
            # l_u = (-mixed_label * output_t).sum(1)
            # task_loss_t = (l_u * label_mask).mean()

            # task loss
            feature_t, _ = model_list[0](image_w)
            output_t_cls = classifier_list[0](feature_t)
            output_t = torch.log_softmax(output_t_cls, dim=1)
            l_u = (-pseudo_label * output_t).sum(1)
            task_loss_t = (l_u * label_mask).mean()

            # Instance-Instance similarity
            feature_sims = []
            sort_idxs = []
            target_sim = []
            target_gcc = []
            with torch.no_grad():
                # calculate the instance2instance similarity, from multiple offline local source domain models (weak augmentation)
                for i in range(1, len(model_list)):
                    # if i == current_domain_index:
                    #     continue
                    temp = F.normalize(model_list[i](image_w)[0])
                    sim = torch.matmul(temp, torch.t(temp))
                    idx = torch.sort(-sim)[1]
                    feature_sims.append(sim)
                    sort_idxs.append(idx)
                    k=12
                    target_tmp = torch.zeros_like(sim)
                    for j in range(image_w.size(0)):
                        target_tmp[j][idx[j][:k]] = 1.0
                    target_sim.append(target_tmp)

                for i in range(len(feature_sims)):
                    if i == 0:
                        target_gcc = feature_sims[i] * graph_weights[i]
                    else:
                        target_gcc += feature_sims[i] * graph_weights[i]
                
                tau = args.t_tau
                target_gcc = target_gcc/tau
                target_gcc = F.softmax(target_gcc, dim=1)

                # calculate the instance2instance similarity, from online local target domain model (weak augmentation)
                local_temp = F.normalize(model_list[0](image_w)[0])
                local_sim = torch.matmul(local_temp, torch.t(local_temp))
                target_local = local_sim/tau
                target_local = F.softmax(target_local, dim=1)
            
            # calculate the instance2instance similarity, from online local target domain model (weak augmentation)
            online_temp = F.normalize(model_list[0](image_w)[0])
            online_sim = torch.matmul(online_temp, torch.t(online_temp))
            online_sim_tmp = logsoftmax(online_sim)

            # inter_irc loss, online local target model (weak augmentation) -> offline local source models (weak augmentation), cross domains
            gcc_loss = online_sim_tmp * target_gcc
            gcc_loss = - gcc_loss.mean(0).sum()

            # calculate the instance2instance similarity, from online local target domain model (strong augmentation)
            online_temp = F.normalize(model_list[0](image_s)[0])
            online_sim = torch.matmul(online_temp, torch.t(online_temp))
            online_sim_tmp_strong = logsoftmax(online_sim)

            # intra_irc loss, online local target model (strong augmentation) -> online local target model (weak augmentation), cross data views
            gcc_local = online_sim_tmp_strong * target_local
            gcc_local = - gcc_local.mean(0).sum()

            # overall losses
            loss = task_loss_t + args.t_inter * gcc_loss + args.t_intra * gcc_local
            loss.backward()
            optimizer_list[0].step()
            classifier_optimizer_list[0].step()
    
    # save the local models, before model aggregation
    pre_models = []
    pre_classifiers = []
    for i in range(0, len(model_list)):
        pre_models.append(copy.deepcopy(model_list[i]))
        pre_classifiers.append(copy.deepcopy(classifier_list[i]))
    
    # test the accuracy of local target domain model, before model aggregation
    target_domain = '******target******'
    if args is not None:
        target_domain = args.target_domain
    acc = test(target_domain, source_domains, test_dloader_list, model_list, classifier_list, epoch, writer, num_classes, states='local')
    
    # aggregating weights of local models: average strategy
    domain_weight = []
    num_domains = len(model_list)
    for i in range(num_domains):
        domain_weight.append(1.0/num_domains)
    
    # model aggregation
    federated_avg(model_list, domain_weight, mode='fedavg')
    federated_avg(classifier_list, domain_weight, mode='fedavg')
    
    return acc, pre_models, pre_classifiers
