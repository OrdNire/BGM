import argparse, math, time, warnings, copy, numpy as np, os.path as path
import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm
import random
import sklearn
import os
import sys
import faiss
import pickle

from sklearn.metrics import average_precision_score
from utils.func import sample_related_index
from models.SpaceIR.DictTree import TreeNode
from models.SpaceIR.InvertedIndex import InvertedIndex
from sklearn.preprocessing import StandardScaler
from evaluate import compute_noise_rate, compute_mAP, compute_ndcg, compute_recall, compute_noise_ndcg
from utils.func import logger_print
import torch.distributed as dist
from custom_loss import cls_weights_conf_loss

def baseAR_inference_for166(
    args,
    model,
    data,
    logger,
    code,
    cfg,
    device
):
    model.eval()
    voc_length = cfg["voc_length"]
    mapping = code["mapping"]
    k_tree = code["dict_tree"]
    for j, batch in enumerate(data):
        img = batch["im"]
        img = img.to(device)

        out = model.search(img, k=args.beam_size, voc_length=voc_length, k_tree=k_tree,
                           ids=torch.tensor(mapping).to(device))

        if j == 0:
            ranks = np.array(out)
        else:
            ranks = np.concatenate((ranks, out), axis=0)

        logger.info('number:{} image \t preds{}'.format(j + 1, out))
    # test
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from evaluate import compute_recall_for166
    metric = {}
    sim_matrix = cfg["sim_matrix"]
    metric["recall"] = compute_recall_for166(sim_matrix, ranks.T)
    logger.info(f"recall: {metric['recall']}")

def baseAR_inference(
    args,
    model,
    data,
    logger,
    code,
    cfg,
    device
):
    model.eval()

    voc_length = cfg["voc_length"]
    mapping = code["mapping"]
    k_tree = code["dict_tree"]

    noise_cnt = 0
    for j, batch in enumerate(data):
        img = batch["im"]
        img = img.to(device)

        out = model.search(img, k=args.beam_size, voc_length=voc_length, k_tree=k_tree,ids=torch.tensor(mapping).to(device))

        if j == 0:
            ranks = np.array(out)
        else:
            ranks = np.concatenate((ranks, out), axis=0)

        logger.info('number:{} image \t preds{}'.format(j+1, out))

        noise_cnt += sum(out >= 3256)

    logger.info(f"return noise cnt by query: {noise_cnt}")
    logger.info(f"return noise cnt: {sum(noise_cnt)}")
    if os.path.isfile(os.path.join("/home/jiangfanzhuo/SpaceIR/data", args.dataset, f"valid_query_idx_{args.beam_size}.pkl")):
        with open(os.path.join("/home/jiangfanzhuo/SpaceIR/data", args.dataset, f"valid_query_idx_{args.beam_size}.pkl"),
                  "rb") as f:
            valid_query_idx = pickle.load(f)
    else:
        valid_query_idx = None
    if valid_query_idx is not None:
        print(f"valid_query: {valid_query_idx} len: {len(valid_query_idx)}")
    ranks = np.asarray(ranks)[valid_query_idx, :].T  # db_size * query
    # test
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from evaluate import compute_mAP, compute_ndcg, compute_accuracy, compute_precision, compute_recall, compute_f1
    metric = {}
    sim_matrix = cfg["sim_matrix"]
    if args.dataset == "MultiScene-Clean":
        match_matrix = (sim_matrix > 0).astype(int)
    else:
        match_matrix = (sim_matrix > 0.5).astype(int)
    # valid correct
    metric["mAP"] = compute_mAP(match_matrix[valid_query_idx, :], ranks)
    metric["nDCG"] = compute_ndcg(sim_matrix[valid_query_idx, :], ranks.shape[0], ranks)
    metric["accuracy"] = compute_accuracy(sim_matrix[valid_query_idx, :], ranks)

    precision_score_matrix = cfg["precision_score_matrix"]
    recall_score_matrix = cfg["recall_score_matrix"]
    f1_score_matrix = cfg["f1_score_matrix"]
    metric["precision"] = compute_precision(precision_score_matrix[valid_query_idx, :], ranks)
    metric["recall"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks)
    metric["f1"] = compute_f1(f1_score_matrix[valid_query_idx, :], ranks)

    logger.info('mAP@{}:{}, nDCG@{}:{} Accuracy@{}:{} precision@{}:{} recall@{}:{} f1@{}:{}'.format(args.beam_size,
                                                                                                    metric["mAP"],
                                                                                                    args.beam_size,
                                                                                                    metric["nDCG"],
                                                                                                    args.beam_size,
                                                                                                    metric["accuracy"],
                                                                                                    args.beam_size,
                                                                                                    metric["precision"],
                                                                                                    args.beam_size,
                                                                                                    metric["recall"],
                                                                                                    args.beam_size,
                                                                                                    metric["f1"]))

def baseAR_train(
    args,
    model,
    data,
    optimizer,
    epoch,
    logger,
    criterion,
    code,
    cfg,
    device,
    scheduler=None,
    stage="indexing"
):
    model.train()
    optimizer.zero_grad()

    voc_length = cfg["voc_length"]
    mapping = code["mapping"]

    match_matrix = cfg["match_matrix"]

    epoch_loss = 0
    for j, batch in enumerate(data):
        img = batch["im"]
        label = batch["onehot_label"]
        idx = batch["idx"]
        if stage == "indexing":
            tgt = np.array([mapping[i] for i in list(idx)])
        elif stage == "retrieval":
            sample_idx = [sample_related_index(match_matrix, i_idx) for i_idx in list(idx)]
            tgt = np.array([mapping[k] for k in sample_idx])
        else:
            raise ValueError("Stage error.")

        tgt[np.where(tgt == -1)] = voc_length
        target = torch.tensor(tgt, dtype=torch.int64)
        img = img.to(device)
        target = target.to(device)

        output = model(img, tgt=target)
        output = output[:, :-1, :]
        output = output.reshape(-1, voc_length)
        target = target.reshape(-1)
        loss = criterion(output, target, args.smoothing)

        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if j % 10 == 0:
            logger.info('Epoch:[{}/{}]\t step={}\t loss={:.10f}\t lr={:.8f}'.format(
                epoch, args.indexing_epochs if stage == "indexing" else args.retrieval_epochs,
                j + 1, loss.item(), optimizer.param_groups[0]['lr']))

        percent = (j + 1) / len(data)
        scheduler.step(epoch + percent)

    scheduler.step(epoch + 1)
    logger.info(f"Epoch:[{epoch}/{args.indexing_epochs if stage == 'indexing' else args.retrieval_epochs}]\t loss={epoch_loss / len(data):.10f}")
    return epoch_loss / len(data)


def tokenizer_train(
        args,
        model,
        label_embedding,
        data,
        optimizer,
        epoch,
        logger,
        criterion_dic,
        scheduler=None):

    classifier_criterion = criterion_dic["classifier_criterion"]
    contrastive_criterion = criterion_dic["contrastive_criterion"]

    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(data):

        image = batch["image"].float()
        noise_image = batch["noise_image"].float()
        combined = torch.cat((image, noise_image), dim=0)
        labels = batch['onehot_label'].float()

        pred, distance_output, int_pred, attns = model(combined.cuda(), None)
        classifier_pred = pred[:image.size(0)]
        classifier_loss = classifier_criterion(classifier_pred.view(labels.size(0), -1), labels.cuda())
        contrastive_loss = contrastive_criterion(classifier_pred.detach(), distance_output[:image.size(0)],
                                                 distance_output[image.size(0):], label_embedding.cuda())

        loss_out = classifier_loss + args.contr_lambda * contrastive_loss

        loss_out.backward()
        # Grad Accumulation
        if ((i + 1) % args.grad_ac_steps == 0):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step(epoch - 1 + (i + 1) / len(data))

        logger.info(f"[{epoch}/{args.epochs}] Total Loss: {loss_out.item()} " +
                    f"Classification Loss: {classifier_loss.item()} " +
                    f"Contrastive Loss: {contrastive_loss.item()}")

@torch.no_grad()
def tokenizer_eval(
        args,
        model,
        label_embedding,
        data,
        epoch,
        logger,
        criterion_dic):

    classifier_criterion = criterion_dic["classifier_criterion"]
    contrastive_criterion = criterion_dic["contrastive_criterion"]

    model.eval()

    # pre-allocate full prediction and target tensors
    all_predictions = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_targets = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_image_ids = []

    for i, batch in enumerate(data):

        image = batch["image"].float()
        noise_image = batch["noise_image"].float()
        combined = torch.cat((image, noise_image), dim=0)
        labels = batch['onehot_label'].float()

        pred, distance_output, int_pred, attns = model(combined.cuda(), None)
        classifier_pred = pred[:image.size(0)]
        classifier_loss = classifier_criterion(classifier_pred.view(labels.size(0), -1), labels.cuda())
        contrastive_loss = contrastive_criterion(classifier_pred.detach(), distance_output[:image.size(0)],
                                                 distance_output[image.size(0):], label_embedding.cuda())

        loss_out = classifier_loss + args.contr_lambda * contrastive_loss

        # update
        start_idx, end_idx = (i * data.batch_size), ((i + 1) * data.batch_size)
        classifier_labels = labels[:image.size(0)]
        if classifier_pred.size(0) != all_predictions[start_idx:end_idx].size(0):
            classifier_pred = classifier_pred.view(classifier_labels.size(0), -1)

        all_predictions[start_idx:end_idx] = classifier_pred.data.cpu()
        all_targets[start_idx:end_idx] = classifier_labels.data.cpu()
        all_image_ids += batch["idx"]

        logger.info(f"[{epoch}/{args.epochs}] Evaluation Total Loss: {loss_out.item()} " +
                    f"Classification Loss: {classifier_loss.item()} " +
                    f"Contrastive Loss: {contrastive_loss.item()}")

    # evaluate
    meanAP = average_precision_score(all_targets, all_predictions, average='macro', pos_label=1)
    logger.info(f"[{epoch}/{args.epochs}] mAP:{meanAP}")
    return meanAP

def SIR_pretrain(
        args,
        model,
        data,
        optimizer,
        epoch,
        logger,
        criterion_dic,
        scheduler=None
):
    classifier_criterion = criterion_dic["classifier_criterion"]

    model.train()
    optimizer.zero_grad()

    batch_loss = 0
    for i, batch in enumerate(data):

        image = batch["im"].float()
        noise_image = batch["noise_im"].float()
        noise_level = batch["noise_level"].float()
        combined = torch.cat((image, noise_image), dim=0)
        labels = batch['onehot_label'].float().cuda()
        classifier_output, _ = model(combined.cuda(), tgt_src=None, tgt=None, label_onehot=None,
                                                   stage="classification")

        # classification loss
        classifier_pred = classifier_output[:image.size(0)]
        classifier_loss = classifier_criterion(classifier_pred.view(labels.size(0), -1), labels)

        loss_out = classifier_loss
        batch_loss += loss_out.item()

        loss_out.backward()
        # Grad Accumulation
        if ((i + 1) % args.grad_ac_steps == 0):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step(epoch - 1 + (i + 1) / len(data))

        if i % 10 == 0:
            logger.info(f"[{epoch}/{args.classification_epochs}] Total Loss: {loss_out.item()} ")
    logger.info(f"[{epoch}/{args.classification_epochs}] Batch Loss: {batch_loss / len(data)}")
    return batch_loss / len(data)

def SIR_classification_train(
        args,
        model,
        data,
        optimizer,
        epoch,
        logger,
        criterion_dic,
        scheduler=None):

    classifier_criterion = criterion_dic["classifier_criterion"]
    contrastive_criterion = criterion_dic["contrastive_criterion"]
    global_contrastive_criterion = criterion_dic["global_contrastive_criterion"]
    ce_loss = criterion_dic["ce_loss"]

    model.train()
    optimizer.zero_grad()

    batch_loss = 0
    for i, batch in enumerate(data):

        image = batch["im"].float()
        noise_image = batch["noise_im"].float()
        noise_level = batch["noise_level"].float()
        combined = torch.cat((image, noise_image), dim=0)
        labels = batch['onehot_label'].float().cuda()
        classifier_output, distance_output = model(combined.cuda(), tgt_src=None, tgt=None, label_onehot=None, stage="classification")

        # classification loss
        classifier_pred = classifier_output[:image.size(0)]
        classifier_loss = classifier_criterion(classifier_pred.view(labels.size(0), -1), labels)

        # contrative loss
        contrastive_loss = contrastive_criterion(classifier_pred.detach(), distance_output[:image.size(0)],
                                                 distance_output[image.size(0):],
                                                 beta=1.0 - noise_level)

        # global feature loss
        # global_contrastive_loss = global_contrastive_criterion(global_embed[image.size(0):], labels)

        loss_out = classifier_loss + args.contr_lambda * contrastive_loss
        # loss_out = global_contrastive_loss
        # loss_out = classifier_loss
        batch_loss += loss_out.item()

        loss_out.backward()
        # Grad Accumulation
        if ((i + 1) % args.grad_ac_steps == 0):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step(epoch - 1 + (i + 1) / len(data))

        if i % 10 == 0:
            logger.info(f"[{epoch}/{args.classification_epochs}] Total Loss: {loss_out.item()} " +
                        f"Classification Loss: {classifier_loss.item()} " +
                        f"Contrastive Loss: {contrastive_loss.item()} ")
            # logger.info(f"[{epoch}/{args.classification_epochs}] Total Loss: {loss_out.item()} ")
    logger.info(f"[{epoch}/{args.classification_epochs}] Batch Loss: {batch_loss/len(data)}")
    return batch_loss/len(data)

@torch.no_grad()
def SIR_classification_eval(
        args,
        model,
        data):

    model.eval()

    pred = torch.empty((0, args.num_labels))
    target = torch.empty((0, args.num_labels))

    for i, batch in enumerate(data):

        image = batch["im"].float()
        labels = batch['onehot_label'].float()
        classifier_output, distance_output = model(image.cuda(), tgt_src=None, tgt=None,
                                                   label_onehot=None, stage="classification")

        # classification loss
        classifier_pred = F.sigmoid(classifier_output)
        pred = torch.cat((pred, classifier_pred.cpu()), dim=0)
        target = torch.cat((target, labels.cpu()), dim=0)

    meanAP = average_precision_score(target, pred, average='macro', pos_label=1)
    print(meanAP)
    return meanAP

def SIR_indexing_train(
        args,
        model,
        data,
        optimizer,
        epoch,
        logger,
        codebook,
        criterion_dic,
        scheduler=None):

    seq2seq_criterion = criterion_dic["seq2seq_criterion"]
    ce_loss = criterion_dic["ce_loss"]
    if codebook is not None:
        clean_codes = torch.tensor(codebook["clean_codes"], dtype=torch.long)
        noise_codes = torch.tensor(codebook["noise_codes"], dtype=torch.long)

    model.train()
    optimizer.zero_grad()

    batch_loss = 0
    for i, batch in enumerate(data):

        image = batch["im"].float()
        idx = batch["idx"].long()
        noise_image = batch["noise_im"].float()
        combined = torch.cat((image, noise_image), dim=0)

        # tgt = torch.cat((clean_codes[idx], noise_codes[idx]), dim=0).cuda()
        dec_output, y, y_id = model(combined.cuda(), tgt_src=None, tgt=None, label_onehot=None, stage="indexing")
        if args.decoder_type == "Transformer":
            loss_out = seq2seq_criterion(dec_output[-1], y.squeeze(1).long())
        elif args.decoder_type == "MLP":
            loss_out = seq2seq_criterion(dec_output, y.squeeze(1).long())

        batch_loss += loss_out.item()

        loss_out.backward()
        # Grad Accumulation
        if ((i + 1) % args.grad_ac_steps == 0):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step(epoch - 1 + (i + 1) / len(data))

        if i % 10 == 0:
            logger.info(f"[{epoch}/{args.indexing_epochs}] Total Loss: {loss_out.item()} ")
    logger.info(f"[{epoch}/{args.indexing_epochs}] Batch Loss: {batch_loss/len(data)}")
    return batch_loss/len(data)

def SIR_retrieval_train(
        args,
        model,
        data,
        optimizer,
        epoch,
        logger,
        codebook,
        criterion_dic,
        scheduler=None):

    seq2seq_criterion = criterion_dic["seq2seq_criterion"]
    ce_loss = criterion_dic["ce_loss"]
    if codebook is not None:
        clean_codes = torch.tensor(codebook["clean_codes"], dtype=torch.long)

    model.train()
    optimizer.zero_grad()

    batch_loss = 0
    for i, batch in enumerate(data):

        image = batch["im"].float().cuda()
        # tgt = clean_codes[tgt_idx].cuda()

        # exp1:
        tgt_image = batch["tgt_im"].float().cuda()
        label = batch["onehot_label"].long().cuda()
        tgt_label = batch["tgt_onehot_label"].long().cuda()
        label_onehot = label & tgt_label

        dec_output, y = model(image, tgt_src=tgt_image, tgt=None, label_onehot=label_onehot.cuda().detach(), stage="retrieval")

        if args.decoder_type == "Transformer":
            loss_out = seq2seq_criterion(dec_output[-1], y.squeeze(1).long())
        elif args.decoder_type == "MLP":
            loss_out = seq2seq_criterion(dec_output, y.squeeze(1).long())

        # loss_out = seq2seq_loss
        batch_loss += loss_out.item()

        loss_out.backward()
        # Grad Accumulation
        if ((i + 1) % args.grad_ac_steps == 0):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step(epoch - 1 + (i + 1) / len(data))

        if i % 10 == 0:
            logger.info(f"[{epoch}/{args.retrieval_epochs}] seq2seq Loss: {loss_out.item()}")
    logger.info(f"[{epoch}/{args.retrieval_epochs}] Batch Loss: {batch_loss/len(data)}")
    return batch_loss / len(data)


def SIR_inference(
        args,
        model,
        label_embedding,
        query_data,
        gallery_data,
        logger,
        cfg):
    dict_tree = TreeNode()
    ividx = InvertedIndex()
    model.eval()

    # # construct id
    # # ps: suppose B = 1
    if os.path.isfile(os.path.join(os.path.dirname(args.model_path), "y_list.pkl")):
        with open(os.path.join(os.path.dirname(args.model_path), "y_list.pkl"), "rb") as f:
            final_y_list_dic = pickle.load(f)
        final_y_list = final_y_list_dic["final_y_list"]
    else:
        final_y_list = []
        save_y_list = {}
        for i, batch in enumerate(tqdm(gallery_data)):
            img = batch["im"].float().cuda()
            idx = batch["idx"].long()
            # y, global_embed = model(img, label_embedding=expand_label_embedding.cuda().detach(), stage="tokenizer") # tgt: (num_labels, 2)
            y, enc_output, _ = model(src=img, stage="tokenizer")
            y_list = y.int().tolist()
            final_y_list.append((int(idx[0].item()), y_list))
        save_y_list["final_y_list"] = final_y_list
        with open(os.path.join(os.path.dirname(args.model_path), "y_list.pkl"), "wb") as f:
            pickle.dump(save_y_list, f)

    for indices, y_list in final_y_list:
        dict_tree.insert_many(y_list)
        ividx_list = [f"{int(pair[0])},{int(pair[1])}" for pair in y_list]
        ividx.add_content(indices, ividx_list)

    ividx.display_index()

    # beam search
    ranks = []
    less_list = []
    valid_query_idx = []
    for i, batch in enumerate(query_data):

        image = batch["im"].float().cuda()
        idx = batch["idx"].long()
        label = batch["onehot_label"].long().cuda()

        expand_label_embedding = label_embedding.unsqueeze(0).repeat(image.shape[0], 1, 1)
        # ans, score = model.module.search_label_guild(image, label, beam_size=args.beam_size, dict_tree=dict_tree, label_embedding=expand_label_embedding.cuda().detach()) # ans: num_labels, beam_size, 2
        # ans, score, global_embed = model.module.search(image, beam_size=args.beam_size, dict_tree=dict_tree, label_embedding=expand_label_embedding.cuda().detach()) # ans: num_labels, beam_size, 2
        ans, score, q_global_embed = model.module.search(image, beam_size=args.beam_size, dict_tree=dict_tree)
        q_idx = batch["idx"].long()
        # ans = ans.permute(1, 0, 2).cpu() # beam_size, num_labels, 2
        # ans_list = [[f"{int(pair[0].item())},{int(pair[1].item())}" for pair in beam] for beam in ans]
        # ans = ividx.search_many(ans_list)[:args.beam_size]
        rank_score, pred = rrank_origin(ans, score, ividx)
        print(f"query idx: {q_idx} score: {rank_score[:args.return_size]}")
        debug(batch, pred, cfg, ividx, args)
        less_list.append(len(pred))

        # logger.info(f"size: {len(final_pred)} pred map: {final_pred[:args.beam_size]}")
        # debug(batch, final_pred[:args.beam_size], cfg)

        if len(pred) >= args.return_size:
            ranks.append(pred[:args.return_size])
            valid_query_idx.append(int(idx[0].item()))

    print(f"count: {np.unique(np.array(less_list))}")
    with open(os.path.join(args.meta_dir, f"valid_query_idx_{str(args.return_size)}.pkl"), "wb") as f:
        pickle.dump(valid_query_idx, f)
    # with open(os.path.join(args.meta_dir, f"valid_query_idx_{str(args.return_size)}.pkl"), "rb") as f:
    #     valid_query_idx = pickle.load(f)
    ranks = np.asarray(ranks).T  # beam_size * query
    print(f"ranks shape: {ranks.shape} valid len: {len(valid_query_idx)}")
    # Test
    from evaluate import compute_mAP, compute_ndcg, compute_accuracy, compute_precision, compute_recall, compute_f1
    metric = {}
    sim_matrix = cfg["sim_matrix"]
    if args.dataset == "MultiScene-Clean":
        match_matrix = (sim_matrix > 0).astype(int)
    else:
        match_matrix = (sim_matrix > 0.5).astype(int)
    # valid correct
    metric["mAP"] = compute_mAP(match_matrix[valid_query_idx, :], ranks)
    metric["nDCG"] = compute_ndcg(sim_matrix[valid_query_idx, :], ranks.shape[0], ranks)
    metric["accuracy"] = compute_accuracy(sim_matrix[valid_query_idx, :], ranks)

    precision_score_matrix = cfg["precision_score_matrix"]
    recall_score_matrix = cfg["recall_score_matrix"]
    f1_score_matrix = cfg["f1_score_matrix"]
    metric["precision"] = compute_precision(precision_score_matrix[valid_query_idx, :], ranks)
    metric["recall"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks)
    metric["f1"] = compute_f1(f1_score_matrix[valid_query_idx, :], ranks)

    logger.info(
        'Origin mAP@{}:{}, nDCG@{}:{} Accuracy@{}:{} precision@{}:{} recall@{}:{} f1@{}:{}'.format(args.return_size,
                                                                                                   metric["mAP"],
                                                                                                   args.return_size,
                                                                                                   metric["nDCG"],
                                                                                                   args.return_size,
                                                                                                   metric["accuracy"],
                                                                                                   args.return_size,
                                                                                                   metric["precision"],
                                                                                                   args.return_size,
                                                                                                   metric["recall"],
                                                                                                   args.return_size,
                                                                                                   metric["f1"]))


@torch.no_grad()
def SIR_inference_exp1(
        args,
        model,
        indexing_model,
        label_embedding,
        query_data,
        gallery_data,
        logger,
        cfg):

    dict_tree = TreeNode()
    ividx = InvertedIndex()
    model.eval()
    if indexing_model is not None:
        indexing_model.eval()

    # # construct id
    # # ps: suppose B = 1
    if os.path.isfile(os.path.join(args.model_path ,"y_list.pkl")):
        with open(os.path.join(args.model_path, "y_list.pkl"), "rb") as f:
            final_y_list_dic = pickle.load(f)
        final_y_list = final_y_list_dic["final_y_list"]
        gallery_global_embed = final_y_list_dic["global_embed"]
    else:
        final_y_list = []
        global_embed_list = []
        gallery_feature_list = []
        save_y_list = {}
        for i, batch in enumerate(tqdm(gallery_data)):
            img = batch["query_im"].float().cuda()
            idx = batch["idx"].long()
            # y, global_embed = model(img, label_embedding=expand_label_embedding.cuda().detach(), stage="tokenizer") # tgt: (num_labels, 2)
            y, enc_output, global_embed = model(src=img, stage="tokenizer")
            # dec_output, tgt, tgt_id = model(src=img, stage="indexing")
            y_list = y.int().tolist()
            final_y_list.append((int(idx[0].item()), y_list))
            global_embed_list.append(global_embed.cpu().numpy())
            # gallery_feature_list.append(F.normalize(enc_output.cpu().detach(), p=2, dim=-1).numpy())
        gallery_global_embed = np.concatenate(global_embed_list, axis=0)
        # gallery_feature = np.concatenate(gallery_feature_list, axis=0)
        save_y_list["final_y_list"] = final_y_list
        save_y_list["global_embed"] = gallery_global_embed
        # save_y_list["gallery_feature"] = gallery_feature
        with open(os.path.join(args.model_path, "y_list.pkl"), "wb") as f:
            pickle.dump(save_y_list, f)

    for indices, y_list in final_y_list:
        dict_tree.insert_many(y_list)
        ividx_list = [f"{int(pair[0])},{int(pair[1])}" for pair in y_list]
        ividx.add_content(indices, ividx_list)

    ividx.display_index()

    # beam search
    ranks = []
    ranks_exp = []
    less_list = []
    valid_query_idx = []
    valid_query_idx_exp = []
    for i, batch in enumerate(query_data):

        image = batch["query_im"].float().cuda()
        idx = batch["idx"].long()
        label = batch["onehot_label"].long().cuda()

        expand_label_embedding = label_embedding.unsqueeze(0).repeat(image.shape[0], 1, 1)
        # ans, score = model.module.search_label_guild(image, label, beam_size=args.beam_size, dict_tree=dict_tree, label_embedding=expand_label_embedding.cuda().detach()) # ans: num_labels, beam_size, 2
        # ans, score, global_embed = model.module.search(image, beam_size=args.beam_size, dict_tree=dict_tree, label_embedding=expand_label_embedding.cuda().detach()) # ans: num_labels, beam_size, 2
        ans, score, q_global_embed = model.module.search(image, beam_size=args.beam_size, dict_tree=dict_tree)
        q_idx = batch["idx"].long()
        # ans = ans.permute(1, 0, 2).cpu() # beam_size, num_labels, 2
        # ans_list = [[f"{int(pair[0].item())},{int(pair[1].item())}" for pair in beam] for beam in ans]
        # ans = ividx.search_many(ans_list)[:args.beam_size]
        rank_score, pred, rank_score_exp, pred_exp = rrank(args, ans, score, ividx, q_global_embed.cpu().numpy(), gallery_global_embed)
        print(f"query idx: {q_idx} score: {rank_score_exp[:args.return_size]}")
        debug(batch, pred_exp, cfg, ividx, args)
        less_list.append(len(pred_exp))

        # logger.info(f"size: {len(final_pred)} pred map: {final_pred[:args.beam_size]}")
        # debug(batch, final_pred[:args.beam_size], cfg)

        if len(pred) >= args.return_size and len(pred_exp) >= args.return_size:
            ranks.append(pred[:args.return_size])
            ranks_exp.append(pred_exp[:args.return_size])
            valid_query_idx.append(int(idx[0].item()))

    print(f"count: {np.unique(np.array(less_list))}")
    with open(os.path.join(args.meta_dir, f"valid_query_idx_{str(args.return_size)}.pkl"), "wb") as f:
        pickle.dump(valid_query_idx, f)

    # 噪声率
    noise_rate = compute_noise_rate(args, cfg, ranks)
    noise_rate_exp = compute_noise_rate(args, cfg, ranks_exp)

    ranks = np.asarray(ranks).T # beam_size * query
    print(f"ranks shape: {ranks.shape} valid len: {len(valid_query_idx)}")
    # Test
    from evaluate import compute_mAP, compute_ndcg, compute_accuracy, compute_precision, compute_recall, compute_f1
    metric = {}
    sim_matrix = cfg["sim_matrix"]
    if args.dataset == "MultiScene-Clean":
        match_matrix = (sim_matrix > 0).astype(int)
    else:
        match_matrix = (sim_matrix > 0.5).astype(int)
    # valid correct
    metric["mAP100"] = compute_mAP(match_matrix[valid_query_idx, :], ranks)
    metric["mAP50"] = compute_mAP(match_matrix[valid_query_idx, :], ranks[:50, :])
    metric["mAP30"] = compute_mAP(match_matrix[valid_query_idx, :], ranks[:30, :])
    metric["mAP10"] = compute_mAP(match_matrix[valid_query_idx, :], ranks[:10, :])
    metric["nDCG"] = compute_ndcg(sim_matrix[valid_query_idx, :], ranks.shape[0], ranks)
    metric["accuracy"] = compute_accuracy(sim_matrix[valid_query_idx, :], ranks)

    precision_score_matrix = cfg["precision_score_matrix"]
    recall_score_matrix = cfg["recall_score_matrix"]
    f1_score_matrix = cfg["f1_score_matrix"]
    metric["precision"] = compute_precision(precision_score_matrix[valid_query_idx, :], ranks)
    metric["recall100"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks)
    metric["recall1"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:1, :])
    metric["recall5"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:5, :])
    metric["recall10"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:10, :])
    metric["recall30"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:30, :])
    metric["recall50"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:50, :])
    metric["f1"] = compute_f1(f1_score_matrix[valid_query_idx, :], ranks)

    logger.info('Origin mAP@{}:{}, nDCG@{}:{} Accuracy@{}:{} precision@{}:{} recall@{}:{} f1@{}:{}'.format(args.return_size, metric["mAP100"],
                                               args.return_size, metric["nDCG"],
                                               args.return_size, metric["accuracy"],
                                               args.return_size, metric["precision"],
                                               args.return_size, metric["recall100"],
                                               args.return_size, metric["f1"]))
    logger.info(f"Origin Noise rate: {noise_rate}")
    logger.info(
        'Origin mAP@10/30/50:{},{},{} recall@1/5/10/30/50:{},{},{},{},{}'.format(metric["mAP10"],metric["mAP30"],metric["mAP50"],
                                                                                 metric["recall1"],metric["recall5"],metric["recall10"],metric["recall30"],metric["recall50"]))

    # exp evaluate
    ranks_exp = np.asarray(ranks_exp).T
    metric = {}
    sim_matrix = cfg["sim_matrix"]
    if args.dataset == "MultiScene-Clean":
        match_matrix = (sim_matrix > 0).astype(int)
    else:
        match_matrix = (sim_matrix > 0.5).astype(int)
    # valid correct
    metric["mAP100"] = compute_mAP(match_matrix[valid_query_idx, :], ranks_exp)
    metric["mAP50"] = compute_mAP(match_matrix[valid_query_idx, :], ranks_exp[:50, :])
    metric["mAP30"] = compute_mAP(match_matrix[valid_query_idx, :], ranks_exp[:30, :])
    metric["mAP10"] = compute_mAP(match_matrix[valid_query_idx, :], ranks_exp[:10, :])
    metric["nDCG"] = compute_ndcg(sim_matrix[valid_query_idx, :], ranks_exp.shape[0], ranks_exp)
    metric["accuracy"] = compute_accuracy(sim_matrix[valid_query_idx, :], ranks_exp)

    precision_score_matrix = cfg["precision_score_matrix"]
    recall_score_matrix = cfg["recall_score_matrix"]
    f1_score_matrix = cfg["f1_score_matrix"]
    metric["precision"] = compute_precision(precision_score_matrix[valid_query_idx, :], ranks_exp)
    metric["recall100"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks_exp)
    metric["recall1"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks_exp[:1, :])
    metric["recall5"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks_exp[:5, :])
    metric["recall10"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks_exp[:10, :])
    metric["recall30"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks_exp[:30, :])
    metric["recall50"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks_exp[:50, :])
    metric["f1"] = compute_f1(f1_score_matrix[valid_query_idx, :], ranks_exp)

    logger.info('Exp mAP@{}:{}, nDCG@{}:{} Accuracy@{}:{} precision@{}:{} recall@{}:{} f1@{}:{}'.format(args.return_size,
                                                                                                    metric["mAP100"],
                                                                                                    args.return_size,
                                                                                                    metric["nDCG"],
                                                                                                    args.return_size,
                                                                                                    metric["accuracy"],
                                                                                                    args.return_size,
                                                                                                    metric["precision"],
                                                                                                    args.return_size,
                                                                                                    metric["recall100"],
                                                                                                    args.return_size,
                                                                                                    metric["f1"]))
    logger.info(f"Exp Noise rate: {noise_rate_exp}")
    logger.info(
        'Exp mAP@10/30/50:{},{},{} recall@1/5/10/30/50:{},{},{},{},{}'.format(metric["mAP10"],metric["mAP30"],metric["mAP50"],
                                                                                 metric["recall1"],metric["recall5"],metric["recall10"],metric["recall30"],metric["recall50"]))

def rrank_origin_wo_constr(ans, score, ividx, cfg):
    # q_embed: (b, embed_dim) g_embed: (gallery_size, embed_dim)
    # beam_size, num_labels, _ = ans.shape
    num_labels, beam_size, _ = ans.shape
    ans_list = [f"{int(pair[0].item())},{int(pair[1].item())}" for l in ans for pair in l]
    pred_list = {str(ans_item): score_item.item() for ans_item, score_item in zip(ans_list, score.view(-1)) if
                 score_item.item() >= 0}
    # union_image = ividx.search_union(pred_list.keys())
    union_image = range(len(cfg["gimlist"]))
    rank_score = {img_id: 0.0 for img_id in union_image}

    for img_id in union_image:
        content = ividx.search_content(img_id)
        valid_content = set(content).intersection(pred_list.keys())
        rank_score[img_id] = sum(pred_list[ct] for ct in valid_content)

    ans_list = torch.unique(ans[:, :, 0]).tolist()
    print(f"ans_list {ans_list} num_labels {num_labels}")
    sorted_rank_score = sorted(rank_score.items(), key=lambda item: (-item[1], len(ividx.content[item[0]])))
    ans = [item[0] for item in sorted_rank_score]

    return sorted_rank_score, ans

def rrank_origin(ans, score, ividx):
    # q_embed: (b, embed_dim) g_embed: (gallery_size, embed_dim)
    # beam_size, num_labels, _ = ans.shape
    num_labels, beam_size, _ = ans.shape
    ans_list = [f"{int(pair[0].item())},{int(pair[1].item())}" for l in ans for pair in l]
    pred_list = {str(ans_item): score_item.item() for ans_item, score_item in zip(ans_list, score.view(-1)) if
                 score_item.item() >= 0}
    union_image = ividx.search_union(pred_list.keys())
    rank_score = {img_id: 0.0 for img_id in union_image}

    for img_id in union_image:
        content = ividx.search_content(img_id)
        valid_content = set(content).intersection(pred_list.keys())
        rank_score[img_id] = sum(pred_list[ct] for ct in valid_content)

    ans_list = torch.unique(ans[:, :, 0]).tolist()
    print(f"ans_list {ans_list} num_labels {num_labels}")
    sorted_rank_score = sorted(rank_score.items(), key=lambda item: (-item[1], len(ividx.content[item[0]])))
    ans = [item[0] for item in sorted_rank_score]

    return sorted_rank_score, ans

def rrank(args, ans, score, ividx, q_embed, g_embed):
    # q_embed: (b, embed_dim) g_embed: (gallery_size, embed_dim)
    # beam_size, num_labels, _ = ans.shape
    num_labels, beam_size, _ = ans.shape
    # q_cls_s = dict(zip(ans[:, 0, 0].tolist(), cls_s.cpu().tolist()))
    ans_list = [f"{int(pair[0].item())},{int(pair[1].item())}" for l in ans for pair in l]
    pred_list = {str(ans_item): score_item.item() for ans_item, score_item in zip(ans_list, score.view(-1)) if score_item.item() >= 0}
    union_image = ividx.search_union(pred_list.keys())
    rank_score = {img_id: 0.0 for img_id in union_image}

    # global feature
    def l2_normalize(array):
        norm = np.linalg.norm(array, ord=2, axis=1, keepdims=True)
        return array / norm

    global_score = (l2_normalize(q_embed)@l2_normalize(g_embed).T)[0]

    # global feature(hash)
    # q_embed = np.sign(q_embed)
    # g_embed = np.sign(g_embed)
    # similarity = (1 - np.dot(q_embed, g_embed.T) / 64)[0] # 1 * db_size
    # sim_ord = np.argsort(similarity, axis=1) # q * db_size

    for img_id in union_image:
        content = ividx.search_content(img_id)
        valid_content = set(content).intersection(pred_list.keys())
        rank_score[img_id] = sum(pred_list[ct] for ct in valid_content)

    ans_list = torch.unique(ans[:,:,0]).tolist()
    ### origin:
    print(f"ans_list {ans_list} num_labels {num_labels}")
    sorted_rank_score = sorted(rank_score.items(), key=lambda item: (-item[1], len(ividx.content[item[0]])))
    # sorted_rank_score = sorted(rank_score.items(),
    #                            key=lambda item: custom_sort_key(item[1], ividx.content[item[0]], ans_list))

    ans = [item[0] for item in sorted_rank_score]

    ### exp: origin + global:
    sorted_rank_score_exp = sorted(rank_score.items(), key=lambda item: (-item[1], len(ividx.content[item[0]]), -global_score[item[0]]))
    ans_exp = [item[0] for item in sorted_rank_score_exp]

    return sorted_rank_score, ans, sorted_rank_score_exp, ans_exp

def debug(batch, pred, cfg, ividx, args):
    q_idx = batch["idx"].long()
    q_label = batch["onehot_label"].long()
    labels_list = [np.where(vector == 1)[0].tolist() for vector in q_label]
    print(f"query idx:{q_idx}; label: {labels_list}")
    gclasses = cfg["gclasses"]
    pred_label_list = []
    for p_idx in pred:
        pred_analyze = [p_idx]
        pred_analyze.append(gclasses[p_idx]["label"])
        pred_analyze.append(ividx.content[p_idx])
        pred_label_list.append(pred_analyze)
    print(f"pred idx, gt_label, pred_id: {pred_label_list[:args.return_size]}")

@torch.no_grad()
def SIR_tokenizer(
        args,
        model,
        label_embedding,
        data,
        logger):
    model.eval()
    clean_features = torch.zeros(len(data.dataset), args.feat_dim).cuda()
    noise_features = torch.zeros(len(data.dataset), args.feat_dim).cuda()
    for i, batch in enumerate(tqdm(data)):
        img = batch["im"].float().cuda()
        noise_image = batch["noise_im"].float()
        idx = batch["idx"].long()
        expand_label_embedding = label_embedding.unsqueeze(0).repeat(img.shape[0], 1, 1)
        clean_feat = model(img, tgt=None, label_embedding=expand_label_embedding.cuda().detach(), stage="tokenizer") # tgt: (num_labels, 2)
        noise_feat = model(noise_image, tgt=None, label_embedding=expand_label_embedding.cuda().detach(), stage="tokenizer") # tgt: (num_labels, 2)
        clean_features[idx] = clean_feat
        noise_features[idx] = noise_feat

    clean_features = clean_features.cpu().numpy()
    noise_features = noise_features.cpu().numpy()
    comb_features = np.concatenate((clean_features, noise_features), axis=0)
    orgin_num, _ = clean_features.shape
    num_vec, dim = comb_features.shape

    # rq
    pq = faiss.ProductQuantizer(dim, 1, args.k_bit)
    x_q = []
    for i in range(args.id_len):
        pq.train(comb_features)
        codes = pq.compute_codes(comb_features)
        if i == 0:
            rq_codes = codes
            codebook = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)
            datarec = pq.decode(codes)
        else:
            rq_codes = np.concatenate((rq_codes, codes), axis=1)
            codebook = np.concatenate((codebook, faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)),
                                      axis=0)
            datarec += pq.decode(codes)
        x_q.append(datarec.copy())
        comb_features -= pq.decode(codes)

    clean_codes = rq_codes[:orgin_num]
    noise_codes = rq_codes[orgin_num:]

    logger.info(f"clean codes: {clean_codes.shape}, {clean_codes}")
    logger.info(f"noise codes: {noise_codes.shape}, {noise_codes}")
    return clean_codes, noise_codes

## EXP2
def phase1_train_SIR2(
        args,
        model,
        data,
        optimizer,
        epoch,
        logger,
        criterion_dic,
        scheduler=None
):
    classifier_criterion = criterion_dic["classifier_criterion"]
    contrastive_criterion = criterion_dic["contrastive_criterion"]
    ce_loss = criterion_dic["ce_loss"]

    model.train()
    optimizer.zero_grad()

    batch_loss = 0
    for i, batch in enumerate(data):

        image = batch["im"].float()
        noise_image = batch["noise_im"].float()
        noise_level = batch["noise_level"].float()
        combined = torch.cat((image, noise_image), dim=0)
        labels = batch['onehot_label'].float().cuda()
        _, classifier_output, distance_output, dec_output, tgt = model(combined.cuda(), stage="phase1")

        # classification loss
        classifier_pred = classifier_output[:image.size(0)]
        classifier_loss = classifier_criterion(classifier_pred.view(labels.size(0), -1), labels)

        # contrative loss
        contrastive_loss = contrastive_criterion(classifier_pred.detach(), distance_output[:image.size(0)],
                                                 distance_output[image.size(0):],
                                                 beta=1.0 - noise_level)

        # noise loss
        noise_loss = ce_loss(dec_output, tgt.long().squeeze(1).detach())

        loss_out = classifier_loss + args.contr_lambda * contrastive_loss + args.contr_lambda * 0.1 * noise_loss
        # loss_out = classifier_loss
        batch_loss += loss_out.item()

        loss_out.backward()
        # Grad Accumulation
        if ((i + 1) % args.grad_ac_steps == 0):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step(epoch - 1 + (i + 1) / len(data))

        if i % 10 == 0:
            logger.info(f"[{epoch}/{args.classification_epochs}] Total Loss: {loss_out.item()} " +
                        f"Classification Loss: {classifier_loss.item()} " +
                        f"Contrastive Loss: {contrastive_loss.item()} " +
                        f"Noise Decoder Loss: {noise_loss.item()}")
    logger.info(f"[{epoch}/{args.classification_epochs}] Batch Loss: {batch_loss/len(data)}")
    return batch_loss/len(data)

def phase2_train_SIR2(
        args,
        model,
        indexing_model,
        data,
        optimizer,
        epoch,
        logger,
        criterion_dic,
        scheduler=None):

    classifier_criterion = criterion_dic["classifier_criterion"]
    ce_loss = criterion_dic["ce_loss"]

    model.train()
    indexing_model.eval()
    optimizer.zero_grad()

    batch_loss = 0
    for i, batch in enumerate(data):

        image = batch["im"].float()
        tgt_image = batch["tgt_im"].float().cuda()
        cls_tgt, indices, tgt = indexing_model(tgt_image.cuda(), stage="phase2") # cls_tgt: B, num_labels
        classifier_output, dec_output = model(image.cuda(), stage="retrieval", indices=indices) # classifier_output: B, num_labels

        # classification loss
        classifier_loss = classifier_criterion(classifier_output, cls_tgt.detach())

        # noise loss
        noise_loss = ce_loss(dec_output, tgt.long().squeeze(1).detach())

        loss_out = classifier_loss + args.contr_lambda * 0.1 * noise_loss
        # loss_out = classifier_loss
        batch_loss += loss_out.item()

        loss_out.backward()
        # Grad Accumulation
        if ((i + 1) % args.grad_ac_steps == 0):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step(epoch - 1 + (i + 1) / len(data))

        if i % 10 == 0:
            logger.info(f"[{epoch}/{args.classification_epochs}] Total Loss: {loss_out.item()} " +
                        f"Retrieval Loss: {classifier_loss.item()} " +
                        f"Noise Decoder Loss: {noise_loss.item()}")
    logger.info(f"[{epoch}/{args.classification_epochs}] Batch Loss: {batch_loss/len(data)}")
    return batch_loss/len(data)

@torch.no_grad()
def phase1_eval_SIR2(
        args,
        model,
        data):

    model.eval()

    pred = torch.empty((0, args.num_labels))
    target = torch.empty((0, args.num_labels))

    for i, batch in enumerate(data):

        image = batch["im"].float()
        labels = batch['onehot_label'].float()
        _, classifier_output, distance_output, dec_output, tgt = model(image.cuda(), stage="phase1")

        # classification loss
        classifier_pred = F.sigmoid(classifier_output)
        pred = torch.cat((pred, classifier_pred.cpu()), dim=0)
        target = torch.cat((target, labels.cpu()), dim=0)

    meanAP = average_precision_score(target, pred, average='macro', pos_label=1)
    return meanAP

@torch.no_grad()
def SIR_inference_SIR2(
        args,
        model,
        indexing_model,
        label_embedding,
        query_data,
        gallery_data,
        logger,
        cfg):

    dict_tree = TreeNode()
    ividx = InvertedIndex()
    model.eval()

    # # construct id
    # # ps: suppose B = 1
    if os.path.isfile(os.path.join(os.path.dirname(args.model_path) ,"y_list.pkl")):
        with open(os.path.join(os.path.dirname(args.model_path), "y_list.pkl"), "rb") as f:
            final_y_list_dic = pickle.load(f)
        final_y_list = final_y_list_dic["final_y_list"]
    else:
        final_y_list = []
        save_y_list = {}
        for i, batch in enumerate(tqdm(gallery_data)):
            img = batch["im"].float().cuda()
            idx = batch["idx"].long()
            # y, global_embed = model(img, label_embedding=expand_label_embedding.cuda().detach(), stage="tokenizer") # tgt: (num_labels, 2)
            _, _, _, y = indexing_model(src=img, stage="phase2")
            y_list = y.int().tolist()
            final_y_list.append((int(idx[0].item()), y_list))
        save_y_list["final_y_list"] = final_y_list
        with open(os.path.join(os.path.dirname(args.model_path), "y_list.pkl"), "wb") as f:
            pickle.dump(save_y_list, f)

    for indices, y_list in final_y_list:
        dict_tree.insert_many(y_list)
        ividx_list = [f"{int(pair[0])},{int(pair[1])}" for pair in y_list]
        ividx.add_content(indices, ividx_list)

    ividx.display_index()

    # beam search
    ranks = []
    less_list = []
    valid_query_idx = []
    for i, batch in enumerate(query_data):

        image = batch["im"].float().cuda()
        idx = batch["idx"].long()

        expand_label_embedding = label_embedding.unsqueeze(0).repeat(image.shape[0], 1, 1)
        # ans, score = model.module.search_label_guild(image, label, beam_size=args.beam_size, dict_tree=dict_tree, label_embedding=expand_label_embedding.cuda().detach()) # ans: num_labels, beam_size, 2
        # ans, score, global_embed = model.module.search(image, beam_size=args.beam_size, dict_tree=dict_tree, label_embedding=expand_label_embedding.cuda().detach()) # ans: num_labels, beam_size, 2
        ans, score = indexing_model.module.search(image, beam_size=args.beam_size, dict_tree=dict_tree)
        q_idx = batch["idx"].long()
        rank_score, pred = rrank_SIR2(ans, score, ividx)
        print(f"query idx: {q_idx} score: {rank_score[:args.return_size]}")
        debug(batch, pred, cfg, ividx, args)
        less_list.append(len(pred))
        if len(pred) >= args.return_size:
            ranks.append(pred[:args.return_size])
            valid_query_idx.append(int(idx[0].item()))
    print(f"count: {np.unique(np.array(less_list))}")
    print(f"valid query: {valid_query_idx} len: {len(valid_query_idx)}")
    with open(os.path.join(args.meta_dir, f"valid_query_idx_{str(args.return_size)}.pkl"), "wb") as f:
        pickle.dump(valid_query_idx, f)

    ranks = np.asarray(ranks).T # beam_size * query
    print(f"ranks shape: {ranks.shape} valid len: {len(valid_query_idx)}")
    # Test
    from evaluate import compute_mAP, compute_ndcg, compute_accuracy, compute_precision, compute_recall, compute_f1
    metric = {}
    sim_matrix = cfg["sim_matrix"]
    if args.dataset == "MultiScene-Clean":
        match_matrix = (sim_matrix > 0).astype(int)
    else:
        match_matrix = (sim_matrix > 0.5).astype(int)
    # valid correct
    metric["mAP"] = compute_mAP(match_matrix[valid_query_idx, :], ranks)
    metric["nDCG"] = compute_ndcg(sim_matrix[valid_query_idx, :], ranks.shape[0], ranks)
    metric["accuracy"] = compute_accuracy(sim_matrix[valid_query_idx, :], ranks)

    precision_score_matrix = cfg["precision_score_matrix"]
    recall_score_matrix = cfg["recall_score_matrix"]
    f1_score_matrix = cfg["f1_score_matrix"]
    metric["precision"] = compute_precision(precision_score_matrix[valid_query_idx, :], ranks)
    metric["recall"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks)
    metric["f1"] = compute_f1(f1_score_matrix[valid_query_idx, :], ranks)

    logger.info('mAP@{}:{}, nDCG@{}:{} Accuracy@{}:{} precision@{}:{} recall@{}:{} f1@{}:{}'.format(args.return_size, metric["mAP"],
                                               args.return_size, metric["nDCG"],
                                               args.return_size, metric["accuracy"],
                                               args.return_size, metric["precision"],
                                               args.return_size, metric["recall"],
                                               args.return_size, metric["f1"]))

def rrank_SIR2(ans, score, ividx):
    # q_embed: (b, embed_dim) g_embed: (gallery_size, embed_dim)
    # beam_size, num_labels, _ = ans.shape
    num_labels, beam_size, _ = ans.shape
    ans_list = [f"{int(pair[0].item())},{int(pair[1].item())}" for l in ans for pair in l]
    pred_list = {str(ans_item): score_item.item() for ans_item, score_item in zip(ans_list, score.view(-1)) if score_item.item() >= 0}
    union_image = ividx.search_union(pred_list.keys())
    rank_score = {img_id: 0.0 for img_id in union_image}

    for img_id in union_image:
        content = ividx.search_content(img_id)
        valid_content = set(content).intersection(pred_list.keys())
        rank_score[img_id] = sum(pred_list[ct] for ct in valid_content)

    ans_list = torch.unique(ans[:,:,0]).tolist()
    print(f"ans_list {ans_list} num_labels {num_labels}")
    sorted_rank_score = sorted(rank_score.items(), key=lambda item: (-item[1], len(ividx.content[item[0]])))
    ans = [item[0] for item in sorted_rank_score]

    return sorted_rank_score, ans

def adjust_temperature(tau: object, iter: object) -> object:
    tau_new = tau
    if iter % 150 == 0:
        tau_new = max(0.1, tau * math.exp(-1e-4 * iter))
    return tau_new

## Joint Learning
def SIR_joint_train(
        args,
        model,
        data,
        optimizer,
        epoch,
        logger,
        criterion_dic,
        scheduler=None,
        stage="warm"):

    classifier_criterion = criterion_dic["classifier_criterion"]
    contrastive_criterion = criterion_dic["contrastive_criterion"]
    seq2seq_criterion = criterion_dic["seq2seq_criterion"]

    model.train()
    optimizer.zero_grad()

    batch_loss = 0

    if stage == "warm":
        for i, batch in enumerate(data):
            image = batch["im"].float()
            noise_image = batch["noise_im"].float()
            noise_level = batch["noise_level"].float()
            combined = torch.cat((image, noise_image), dim=0)
            labels = batch['onehot_label'].float().cuda()
            classifier_output, distance_output = model(combined.cuda(), tgt_src=None, tgt=None, label_onehot=None,
                                                       stage="classification")
            # classification loss
            classifier_pred = classifier_output[:image.size(0)]
            classifier_loss = classifier_criterion(classifier_pred.view(labels.size(0), -1), labels)
            # contrative loss
            contrastive_loss = contrastive_criterion(classifier_pred.detach(), distance_output[:image.size(0)],
                                                     distance_output[image.size(0):],
                                                     beta=1.0 - noise_level)
            loss_out = classifier_loss + args.contr_lambda * contrastive_loss
            batch_loss += loss_out.item()

            loss_out.backward()
            # Grad Accumulation
            if ((i + 1) % args.grad_ac_steps == 0):
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step(epoch - 1 + (i + 1) / len(data))

            if i % 10 == 0:
                logger.info(f"[{epoch}/{args.warm_epochs}] Total Loss: {loss_out.item()} " +
                            f"Classification Loss: {classifier_loss.item()} " +
                            f"Contrastive Loss: {contrastive_loss.item()} ")
                # logger.info(f"[{epoch}/{args.classification_epochs}] Total Loss: {loss_out.item()} ")
        logger.info(f"[{epoch}/{args.warm_epochs}] Batch Loss: {batch_loss / len(data)}")
    elif stage == "train":

        model_to_use = model.module if hasattr(model, 'module') else model
        for i, batch in enumerate(data):
            # adjust tau
            model_to_use.tau = adjust_temperature(model_to_use.tau, len(data) * (epoch - 1) + i + 1)
            # <q, t_img, t_img_noise>
            image = batch["im"].float().cuda()
            noise_image = batch["noise_im"].float().cuda()
            noise_level = batch["noise_level"].float()
            tgt_image = batch["tgt_im"].float().cuda()
            labels = batch['onehot_label'].long().cuda()
            tgt_label = batch["tgt_onehot_label"].long().cuda()
            label_onehot = labels & tgt_label
            batch_N = image.size(0)
            src_cls_out, distance_src_out, distance_noise_out, dec_output, combine_tgt = model(src=image, noise_src=noise_image, tgt_src=tgt_image, label_onehot=label_onehot, stage="joint_learningV2")
            # classification loss
            classifier_loss = classifier_criterion(src_cls_out.view(batch_N, -1), labels.float())

            # contrative loss
            contrastive_loss = contrastive_criterion(src_cls_out.detach(), distance_src_out,
                                                     distance_noise_out,
                                                     beta=1.0 - noise_level)

            # generative loss
            if args.decoder_type == "Transformer":
                generative_loss = seq2seq_criterion(dec_output[-1], combine_tgt.long())
            elif args.decoder_type == "MLP":
                generative_loss = seq2seq_criterion(dec_output, combine_tgt.long())

            loss_out = classifier_loss + args.contr_lambda * (contrastive_loss + generative_loss)
            batch_loss += loss_out.item()

            loss_out.backward()
            # Grad Accumulation
            if ((i + 1) % args.grad_ac_steps == 0):
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step(epoch - 1 + (i + 1) / len(data))

            if i % 10 == 0:
                logger.info(f"[{epoch}/{args.epochs}] Current temperature: {model_to_use.tau}")
                logger.info(f"[{epoch}/{args.epochs}] Total Loss: {loss_out.item()} " +
                            f"Classification Loss: {classifier_loss.item()} " +
                            f"Contrastive Loss: {contrastive_loss.item()} " +
                            f"Generative Loss: {generative_loss.item()}" )
                # logger.info(f"[{epoch}/{args.classification_epochs}] Total Loss: {loss_out.item()} ")
        logger.info(f"[{epoch}/{args.epochs}] Batch Loss: {batch_loss/len(data)}")
    return batch_loss/len(data)

def SIR_joint_train_w_noise(
        args,
        model,
        data,
        optimizer,
        epoch,
        logger,
        criterion_dic,
        scheduler=None,
        stage="warm"):

    classifier_criterion = criterion_dic["classifier_criterion"]
    contrastive_criterion = criterion_dic["contrastive_criterion"]
    seq2seq_criterion = criterion_dic["seq2seq_criterion"]

    model.train()
    optimizer.zero_grad()

    batch_loss = 0

    if stage == "warm":
        for i, batch in enumerate(data):
            image = batch["query_im"].float()
            noise_image = batch["query_noise_im"].float()
            noise_level = batch["query_noise_level"].float()
            combined = torch.cat((image, noise_image), dim=0)
            labels = batch['onehot_label'].float().cuda()
            classifier_output, distance_output = model(combined.cuda(), tgt_src=None, tgt=None, label_onehot=None,
                                                       stage="classification")
            # classification loss
            classifier_pred = classifier_output[:image.size(0)]
            classifier_loss = classifier_criterion(classifier_pred.view(labels.size(0), -1), labels)
            # contrative loss
            contrastive_loss = contrastive_criterion(classifier_pred.detach(), distance_output[:image.size(0)],
                                                     distance_output[image.size(0):],
                                                     beta=1.0 - noise_level)
            loss_out = classifier_loss + args.contr_lambda * contrastive_loss
            batch_loss += loss_out.item()

            loss_out.backward()
            # Grad Accumulation
            if ((i + 1) % args.grad_ac_steps == 0):
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step(epoch - 1 + (i + 1) / len(data))

            if i % 10 == 0:
                logger.info(f"[{epoch}/{args.warm_epochs}] Total Loss: {loss_out.item()} " +
                            f"Classification Loss: {classifier_loss.item()} " +
                            f"Contrastive Loss: {contrastive_loss.item()} ")
                # logger.info(f"[{epoch}/{args.classification_epochs}] Total Loss: {loss_out.item()} ")
        logger.info(f"[{epoch}/{args.warm_epochs}] Batch Loss: {batch_loss / len(data)}")
    elif stage == "train":

        model_to_use = model.module if hasattr(model, 'module') else model
        for i, batch in enumerate(data):
            # adjust tau
            model_to_use.tau = adjust_temperature(model_to_use.tau, len(data) * (epoch - 1) + i + 1)
            # <query_im, query_noise_im, query_noise_level, tgt_im, tgt_noise_level>
            image = batch["query_im"].float().cuda()
            noise_image = batch["query_noise_im"].float().cuda()
            noise_level = batch["query_noise_level"].float()
            tgt_image = batch["tgt_im"].float().cuda()
            tgt_noise_level = batch["tgt_noise_level"].float().cuda() # (B, )
            labels = batch['onehot_label'].long().cuda()
            tgt_label = batch["tgt_onehot_label"].long().cuda()
            label_onehot = labels & tgt_label
            batch_N = image.size(0)
            src_cls_out, distance_src_out, distance_noise_out, dec_output, combine_tgt, combine_weights = model(src=image, noise_src=noise_image, tgt_src=tgt_image, label_onehot=label_onehot, tgt_noise_level=tgt_noise_level, stage="joint_learningV3")
            # classification loss
            classifier_loss = classifier_criterion(src_cls_out.view(batch_N, -1), labels.float())

            # contrative loss
            contrastive_loss = contrastive_criterion(src_cls_out.detach(), distance_src_out,
                                                     distance_noise_out,
                                                     beta=1.0 - noise_level)

            # generative loss
            if args.decoder_type == "Transformer":
                generative_loss = seq2seq_criterion(dec_output[-1], combine_tgt.long(), combine_weights)
            elif args.decoder_type == "MLP":
                generative_loss = seq2seq_criterion(dec_output, combine_tgt.long(), combine_weights)

            loss_out = classifier_loss + args.contr_lambda * (contrastive_loss + generative_loss)
            batch_loss += loss_out.item()

            loss_out.backward()
            # Grad Accumulation
            if ((i + 1) % args.grad_ac_steps == 0):
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step(epoch - 1 + (i + 1) / len(data))

            if i % 10 == 0:
                logger.info(f"[{epoch}/{args.epochs}] Current temperature: {model_to_use.tau}")
                logger.info(f"[{epoch}/{args.epochs}] Total Loss: {loss_out.item()} " +
                            f"Classification Loss: {classifier_loss.item()} " +
                            f"Contrastive Loss: {contrastive_loss.item()} " +
                            f"Generative Loss: {generative_loss.item()}" )
                # logger.info(f"[{epoch}/{args.classification_epochs}] Total Loss: {loss_out.item()} ")
        logger.info(f"[{epoch}/{args.epochs}] Batch Loss: {batch_loss/len(data)}")
    return batch_loss/len(data)

@torch.no_grad()
def middle_eval(
        args,
        model,
        query_data,
        gallery_data,
        logger,
        cfg):

    dict_tree = TreeNode()
    ividx = InvertedIndex()
    model.eval()

    # # construct id
    # # ps: suppose B = 1
    final_y_list = []
    for i, batch in enumerate(tqdm(gallery_data)):
        img = batch["im"].float().cuda()
        idx = batch["idx"].long()
        y, enc_output, global_embed = model(src=img, stage="tokenizer")
        y_list = y.int().tolist()
        final_y_list.append((int(idx[0].item()), y_list))
    for indices, y_list in final_y_list:
        dict_tree.insert_many(y_list)
        ividx_list = [f"{int(pair[0])},{int(pair[1])}" for pair in y_list]
        ividx.add_content(indices, ividx_list)

    ividx.display_index()

    # beam search
    ranks = []
    valid_query_idx = []
    for i, batch in enumerate(query_data):

        image = batch["im"].float().cuda()
        idx = batch["idx"].long()
        ans, score, q_global_embed = model.module.search(image, beam_size=args.return_size, dict_tree=dict_tree)
        rank_score, pred = rrank_origin(ans, score, ividx)
        if len(pred) >= args.return_size:
            ranks.append(pred[:args.return_size])
            valid_query_idx.append(int(idx[0].item()))

    ranks = np.asarray(ranks).T # beam_size * query
    # Test
    from evaluate import compute_mAP, compute_ndcg, compute_accuracy, compute_precision, compute_recall, compute_f1
    metric = {}
    sim_matrix = cfg["sim_matrix"]
    if args.dataset == "MultiScene-Clean":
        match_matrix = (sim_matrix > 0).astype(int)
    else:
        match_matrix = (sim_matrix > 0.5).astype(int)
    # valid correct
    metric["mAP"] = compute_mAP(match_matrix[valid_query_idx, :], ranks)
    metric["nDCG"] = compute_ndcg(sim_matrix[valid_query_idx, :], ranks.shape[0], ranks)
    metric["accuracy"] = compute_accuracy(sim_matrix[valid_query_idx, :], ranks)
    precision_score_matrix = cfg["precision_score_matrix"]
    recall_score_matrix = cfg["recall_score_matrix"]
    f1_score_matrix = cfg["f1_score_matrix"]
    metric["precision"] = compute_precision(precision_score_matrix[valid_query_idx, :], ranks)
    metric["recall"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks)
    metric["f1"] = compute_f1(f1_score_matrix[valid_query_idx, :], ranks)
    logger.info('Origin mAP@{}:{}, nDCG@{}:{} Accuracy@{}:{} precision@{}:{} recall@{}:{} f1@{}:{}'.format(args.return_size, metric["mAP"],
                                               args.return_size, metric["nDCG"],
                                               args.return_size, metric["accuracy"],
                                               args.return_size, metric["precision"],
                                               args.return_size, metric["recall"],
                                               args.return_size, metric["f1"]))
    return metric["f1"]

## Multi-Modal Learning
def SIRMM_train(
        args,
        model,
        data,
        optimizer,
        epoch,
        logger,
        criterion_dic,
        scheduler=None,
        stage="train"):

    # CLS Loss: Image(Image Indexing), Text(Text2Image Retrieval) # 1-dim id
    classifier_criterion = criterion_dic["classifier_criterion"]
    # SIM Loss: SIM indexer Learning
    contrastive_criterion = criterion_dic["contrastive_criterion"]
    # SIM Loss: Image(Image Indexing), Text(Text2Image Retrieval) # 2-dim id
    seq2seq_criterion = criterion_dic["seq2seq_criterion"]
    contrastive_hardway_criterion = criterion_dic["contrastive_hardway_criterion"]

    optimizer.zero_grad()

    batch_loss = 0

    if stage == "train":
        model.train()
        for i, batch in enumerate(data):
            if args.abl_setting == "hard_way":
                model_to_use = model.module if hasattr(model, 'module') else model
                model_to_use.tau = adjust_temperature(model_to_use.tau, len(data) * (epoch - 1) + i + 1)
                if i % 10 == 0:
                    logger_print(args.rank, logger, f"Current tau: {model_to_use.tau}")
            query = batch["cap"]
            query = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in query.items()}
            tgt_image = batch["im"].to(args.device).float()
            tgt_noise_image = batch["noise_im"].to(args.device).float()
            noise_level = batch["noise_level"].to(args.device).float()
            labels = batch['label'].to(args.device).float()

            if args.abl_setting == "ind":
                tgt_cls_out = model(query, tgt_image, stage="abl_ind", device=args.device)
                img_cls_loss = classifier_criterion(tgt_cls_out.view(labels.size(0), -1), labels)
                loss_out = img_cls_loss
                if i % 10 == 0:
                    logger_print(args.rank, logger, f"[{epoch}/{args.epochs}] Total Loss: {loss_out.item()} " +
                                 f"Img classification Loss: {img_cls_loss.item()} ")
            elif args.abl_setting == "ret":
                txt_cls_out = model(query, stage="abl_ret", device=args.device)
                txt_cls_loss = classifier_criterion(txt_cls_out.view(labels.size(0), -1), labels)
                loss_out = txt_cls_loss
                if i % 10 == 0:
                    logger_print(args.rank, logger, f"[{epoch}/{args.epochs}] Total Loss: {loss_out.item()} " +
                                 f"Text classification Loss: {txt_cls_loss.item()} ")
            elif args.abl_setting == "ind_ret":
                tgt_cls_out, txt_cls_out = model(query, tgt_image, stage="abl_ind_ret", device=args.device)
                img_cls_loss = classifier_criterion(tgt_cls_out.view(labels.size(0), -1), labels)
                txt_cls_loss = classifier_criterion(txt_cls_out.view(labels.size(0), -1), labels)
                loss_out = img_cls_loss + txt_cls_loss
                if i % 10 == 0:
                    logger_print(args.rank, logger, f"[{epoch}/{args.epochs}] Total Loss: {loss_out.item()} " +
                                 f"Image classification Loss: {img_cls_loss.item()} " +
                                 f"Text classification Loss: {txt_cls_loss.item()} ")
            elif args.abl_setting == "hard_way":
                tgt_cls_out, text_cls_out, tgt_sim_out, noise_tgt_sim_out, y = model(query, tgt_image, tgt_noise_image,
                                                                                  stage="abl_hard_way",
                                                                                  device=args.device)
                # CLS Loss
                img_cls_loss = classifier_criterion(tgt_cls_out.view(labels.size(0), -1), labels)
                text_cls_loss = classifier_criterion(text_cls_out.view(labels.size(0), -1), labels)
                # SIM Loss
                noise_level_expand = noise_level.repeat(1, args.num_labels).reshape(-1)
                # contrastive_loss = contrastive_hardway_criterion(tgt_sim_out, noise_tgt_sim_out, label_emb_out, beta=1.0 - noise_level_expand[mask_reshape.detach()])
                contrastive_loss = contrastive_criterion(y.detach(), tgt_sim_out, noise_tgt_sim_out,
                                                         beta=1.0 - noise_level)
                loss_out = img_cls_loss + text_cls_loss + args.contr_lambda * contrastive_loss
                if i % 10 == 0:
                    logger_print(args.rank, logger, f"[{epoch}/{args.epochs}] Total Loss: {loss_out.item()} " +
                                 f"Img classification Loss: {img_cls_loss.item()} " +
                                 f"Text classification Loss: {text_cls_loss.item()} " +
                                 f"Contrastive Loss: {contrastive_loss.item()}")
            elif args.abl_setting == "MT":
                tgt_cls_out, text_cls_out, tgt_sim_out, noise_tgt_sim_out = model(query, tgt_image, tgt_noise_image,
                                                                                     stage="abl_MT",
                                                                                     device=args.device)
                # CLS Loss
                img_cls_loss = classifier_criterion(tgt_cls_out.view(labels.size(0), -1), labels)
                text_cls_loss = classifier_criterion(text_cls_out.view(labels.size(0), -1), labels)
                # SIM Loss
                noise_level_expand = 1.0 - noise_level.repeat(1, args.num_labels).reshape(-1)
                # contrastive_loss = contrastive_hardway_criterion(tgt_sim_out, noise_tgt_sim_out, label_emb_out, beta=1.0 - noise_level_expand[mask_reshape.detach()])
                contrastive_loss = contrastive_criterion(torch.ones((tgt_sim_out.shape[0], args.num_labels),device=args.device), tgt_sim_out, noise_tgt_sim_out,
                                                         beta=1.0 - noise_level)
                loss_out = img_cls_loss + text_cls_loss + args.contr_lambda * contrastive_loss
                if i % 10 == 0:
                    logger_print(args.rank, logger, f"MT [{epoch}/{args.epochs}] Total Loss: {loss_out.item()} " +
                                 f"Img classification Loss: {img_cls_loss.item()} " +
                                 f"Text classification Loss: {text_cls_loss.item()} " +
                                 f"Contrastive Loss: {contrastive_loss.item()}")
            elif args.abl_setting == "PP":
                tgt_cls_out, text_cls_out, tgt_sim_out, noise_tgt_sim_out, y, conf_out, conf_label = model(query, tgt_image, tgt_noise_image,
                                                                                     stage="PP",
                                                                                     device=args.device)
                # CLS Loss
                img_cls_loss = classifier_criterion(tgt_cls_out.view(labels.size(0), -1), labels)
                text_cls_loss = classifier_criterion(text_cls_out.view(labels.size(0), -1), labels)
                # SIM Loss
                noise_level_expand = noise_level.repeat(1, args.num_labels).reshape(-1)
                # contrastive_loss = contrastive_hardway_criterion(tgt_sim_out, noise_tgt_sim_out, label_emb_out, beta=1.0 - noise_level_expand[mask_reshape.detach()])
                contrastive_loss = contrastive_criterion(y.detach(), tgt_sim_out, noise_tgt_sim_out,
                                                         beta=1.0 - noise_level)
                # Conf Loss
                conf_loss = cls_weights_conf_loss(conf_out, conf_label.detach().long(), y.detach())
                loss_out = img_cls_loss + text_cls_loss + args.contr_lambda * (contrastive_loss + conf_loss)
                if i % 10 == 0:
                    logger_print(args.rank, logger, f"PP [{epoch}/{args.epochs}] Total Loss: {loss_out.item()} " +
                                 f"Img classification Loss: {img_cls_loss.item()} " +
                                 f"Text classification Loss: {text_cls_loss.item()} " +
                                 f"Contrastive Loss: {contrastive_loss.item()} " +
                                 f"Conf Loss: {conf_loss.item()}")
            else:
                tgt_cls_out, text_cls_out, tgt_sim_out, noise_tgt_sim_out = model(query, tgt_image, tgt_noise_image,
                                                                                  stage="text_only_joint_learning",
                                                                                  device=args.device)
                # CLS Loss
                img_cls_loss = classifier_criterion(tgt_cls_out.view(labels.size(0), -1), labels)
                text_cls_loss = classifier_criterion(text_cls_out.view(labels.size(0), -1), labels)
                # SIM Loss
                contrastive_loss = contrastive_criterion(tgt_cls_out.detach(), tgt_sim_out, noise_tgt_sim_out,
                                                         beta=1.0 - noise_level)
                loss_out = img_cls_loss + text_cls_loss + args.contr_lambda * contrastive_loss

                if i % 10 == 0:
                    logger_print(args.rank, logger, f"[{epoch}/{args.epochs}] Total Loss: {loss_out.item()} " +
                                 f"Img classification Loss: {img_cls_loss.item()} " +
                                 f"Text classification Loss: {text_cls_loss.item()} " +
                                 f"Contrastive Loss: {contrastive_loss.item()}")

            batch_loss += loss_out.item()

            loss_out.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step(epoch - 1 + (i + 1) / len(data))

        logger_print(args.rank, logger, f"[{epoch}/{args.epochs}] Batch Loss: {batch_loss / len(data)}")
    elif stage == "memory":
        model.train()
        for i, batch in enumerate(data):
            tgt_image = batch["im"].to(args.device).float()
            labels = batch['label'].to(args.device).float()

            tgt_cls_out = model(tgt=tgt_image, stage="image_memory")
            # CLS Loss
            img_cls_loss = classifier_criterion(tgt_cls_out.view(labels.size(0), -1), labels)
            loss_out = img_cls_loss
            batch_loss += loss_out.item()

            loss_out.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step(epoch - 1 + (i + 1) / len(data))

            if i % 10 == 0:
                logger_print(args.rank, logger, f"[{epoch}/{args.mem_epochs}] Total Loss: {loss_out.item()} ")
                # logger.info(f"[{epoch}/{args.classification_epochs}] Total Loss: {loss_out.item()} ")
        logger_print(args.rank, logger, f"[{epoch}/{args.mem_epochs}] Batch Loss: {batch_loss / len(data)}")
    elif stage == "eval":
        batch_loss_img_cls = 0
        batch_loss_txt_cls = 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data):
                query = batch["cap"]
                query = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in query.items()}
                tgt_image = batch["im"].to(args.device).float()
                tgt_noise_image = batch["noise_im"].to(args.device).float()
                noise_level = batch["noise_level"].to(args.device).float()
                labels = batch['label'].to(args.device).float()

                tgt_cls_out, text_cls_out, tgt_sim_out, noise_tgt_sim_out = model(query, tgt_image, tgt_noise_image, stage="text_only_joint_learning")
                # CLS Loss
                img_cls_loss = classifier_criterion(tgt_cls_out.view(labels.size(0), -1), labels)
                text_cls_loss = classifier_criterion(text_cls_out.view(labels.size(0), -1), labels)
                # SIM Loss
                contrastive_loss = contrastive_criterion(tgt_cls_out.detach(), tgt_sim_out, noise_tgt_sim_out,
                                                         beta=1.0 - noise_level)

                loss_out = img_cls_loss + text_cls_loss + args.contr_lambda * contrastive_loss
                batch_loss += loss_out.item()
                batch_loss_img_cls += img_cls_loss.item()
                batch_loss_txt_cls += text_cls_loss.item()
                if i % 10 == 0:
                    logger_print(args.rank, logger, f"Val [{epoch}/{args.epochs}] Total Loss: {loss_out.item()} " +
                                 f"Img classification Loss: {img_cls_loss.item()} " +
                                 f"Text classification Loss: {text_cls_loss.item()} " +
                                 f"Contrastive Loss: {contrastive_loss.item()}")
            logger_print(args.rank, logger, f"Val [{epoch}/{args.epochs}] Batch Loss: {batch_loss / len(data)} Batch img cls: {batch_loss_img_cls/ len(data)} Batch txt cls: {batch_loss_txt_cls/ len(data)}")
    return batch_loss/len(data)

def debugV2(batch, union_image, gallery_dataset):
    q_label = batch["label"]
    print(f"query label: {q_label}")
    pred_labels = []
    for img_id, score in union_image:
        pred_label = gallery_dataset[int(img_id)]["label"]
        pred_labels.append(pred_label)
    print(f"pred label: {pred_labels}")

@torch.no_grad()
def SIRMM_eval(
        args,
        model,
        query_data,
        gallery_data,
        logger,
        stage="train"
):
    local_rank = None
    if stage != "inference":
        local_rank = dist.get_rank()
    ividx = InvertedIndex()
    model.eval()
    # Gallery
    logger_print(local_rank, logger, "Compute Gallery.")
    id_list = []
    for i, batch in enumerate(tqdm(gallery_data)):
        img = batch["im"].to(args.device).float()
        idx = batch["idx"].int()
        identifier, embed = model(query=img, stage="tokenizer", device=args.device) # (N, 2)
        id_list.append((int(idx[0].item()), identifier.tolist()))


    for idx, identifier in id_list:
        content_list = [(str(int(pair[0])), float(pair[1])) for pair in identifier]
        ividx.add_contentV2(idx, content_list)
    if stage == "inference":
        ividx.display_index()

    # valid query i % valid query idx
    valid_use_query_i = []

    # query search
    logger_print(local_rank, logger, "Compute Query.")
    rank_list = []
    query_idx = []
    for i, batch in enumerate(tqdm(query_data)):
        query = batch["cap"]
        query = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in query.items()}
        idx = batch["idx"]
        ans = model.searchV2(args, query)

        pred = [str(l) for l in ans.cpu().tolist()]
        union_image = ividx.search_union(pred)
        if len(union_image) < args.max_topK:
            continue
        rank_score = {}

        # debug
        # debugV2(batch, union_image, gallery_data.dataset)

        for img_id, score in union_image:
            if img_id in rank_score:
                rank_score[img_id] += score
            else:
                rank_score[img_id] = score
        sorted_rank_score = sorted(rank_score.items(), key=lambda item: (-item[1], -len(ividx.content[item[0]])))
        rk = [item[0] for item in sorted_rank_score][:args.max_topK]
        if len(rk) < args.max_topK:
            continue
        rank_list.append(rk)
        query_idx.append(idx[0].item())
        valid_use_query_i.append(i)

    if stage == "inference" and (args.abl_setting=="hard_way"):
        # Save valid query i&idx
        with open(os.path.join("./data", args.dataset, f"valid_query_idx_{str(args.max_topK)}_T2I.pkl"), "wb") as f:
            pickle.dump(valid_use_query_i, f)

        with open(os.path.join("./data", args.dataset, f"valid_query_idx_{str(args.max_topK)}.pkl"), "wb") as f:
            pickle.dump(query_idx, f)


    # Val
    logger_print(local_rank, logger, "Start evaluate.")
    rank = np.asarray(rank_list) # q, max_topK
    print(f"rank shape: {rank.shape}")
    if stage != "inference":
        topK_list = [1, 5, args.max_topK]
    else:
        topK_list = [1, 5, 10, args.max_topK]
        # topK_list = [1, 5, args.max_topK]
    map_matrix = args.map_matrix[query_idx, :]
    ndcg_matrix = args.ndcg_matrix[query_idx, :]

    def compute_avg_score(ranks, test_matrix):
        query_size, db_size = ranks.shape
        return_score = []
        for i in range(query_size):
            score = 0
            i_ranks = ranks[i, :]
            for rk in i_ranks:
                score += test_matrix[i, rk]
            return_score.append(score / len(i_ranks))
        return np.average(return_score)

    map_list = []
    acg_list = []
    ndcg_list = []
    for k in tqdm(topK_list):
        rank_k = rank[:, :k]
        map_list.append(compute_mAP(map_matrix, rank_k.T))
        acg_list.append(compute_avg_score(rank_k, ndcg_matrix))
        ndcg_list.append(compute_ndcg(ndcg_matrix, k, rank_k.T))
    logger_print(local_rank, logger, f"map list: {map_list}")
    logger_print(local_rank, logger, f"acg: {acg_list}")
    logger_print(local_rank, logger, f"nDCG: {ndcg_list}")
    return np.mean(map_list + acg_list + ndcg_list)

def case_study(rk_matrix, query_dataset, gallery_dataset):
    with open("SIR_case_study.txt", "w") as file:
        for q_i, rk in enumerate(rk_matrix):
            q_label = query_dataset.db[q_i]["label"]
            cap = query_dataset.db[q_i]["cap"]
            q_id = np.where(q_label == 1)[0].tolist()
            file.write(f"query:{q_i} sample:{query_dataset.db[q_i]['idx']} id: {q_id} cap: {cap}\n")
            for r in rk[:5]:
                r_label = gallery_dataset.db[r]["label"]
                r_id = np.where(r_label == 1)[0].tolist()
                over_number = np.sum(q_label * r_label)
                file.write(f"id:{r_id} n:{over_number} level:{gallery_dataset.imgid2level[r]} file:{str(gallery_dataset.db[r]['im']).split('/')[-1]}; ")
            file.write(f"\n")


@torch.no_grad()
def SIRMM_eval_CGNoise(
        args,
        model,
        query_data,
        gallery_data,
        logger
):
    ividx = InvertedIndex()
    model.eval()
    # Gallery
    logger.info("Compute Gallery.")
    id_list = []

    # with open('efficiency_exp/tmp_list.pkl', 'rb') as f:
    #     id_list = pickle.load(f)
    test_gallery_np = []
    test_gallery_noise_level = []
    for i, batch in enumerate(tqdm(gallery_data)):
        img = batch["im"].to(args.device).float()
        idx = batch["idx"].int()
        if args.abl_setting == "PP":
            identifier, embed = model(query=img, stage="tokenizer_PP", device=args.device)
        else:
            identifier, embed = model(query=img, stage="tokenizer", device=args.device) # (N, 2)
        # identifier, embed = model(query=img, stage="tokenizer", device=args.device)  # (N, 2)
        id_list.append((int(idx[0].item()), identifier.tolist()))
        test_gallery_np.append(embed.cpu().numpy())
        test_gallery_noise_level.append(gallery_data.dataset.imgid2level[int(idx[0].item())])

    vis_study_dict = {"gallery_embd": np.concatenate(test_gallery_np, axis=0), # N, num_labels, dim
                      "noise_level": np.asarray(test_gallery_noise_level)}
    with open(f"efficiency_exp/vis_study_{args.dataset}.pkl", "wb") as f:
        pickle.dump(vis_study_dict, f)

    exit(0)
    #
    # with open("efficiency_exp/tmp_list.pkl", 'wb') as f:
    #     pickle.dump(id_list, f)
    # exit(0)
    # int_list = []
    # float_list = []

    for idx, identifier in id_list:
        # for j in range(9):
        content_list = [(str(int(pair[0])), float(pair[1])) for pair in identifier]
        # int_list.extend([int(pair[0]) for pair in identifier])
        # float_list.extend([float(pair[1]) for pair in identifier])
        ividx.add_contentV2(idx, content_list)

    # np.save(f"efficiency_exp/int_f.npy", np.asarray(int_list, dtype=np.int8))
    # np.save(f"efficiency_exp/float_f.npy", np.asarray(float_list, dtype=np.float32))

    ividx.display_index()

    # valid query i % valid query idx
    valid_use_query_i = []

    # query search
    logger.info("Compute Query.")
    rank_list = []
    rank_idx_list = []
    query_idx = []
    imgid2level = gallery_data.dataset.imgid2level

    time_list = []
    for i, batch in enumerate(tqdm(query_data)):
        query = batch["cap"]
        query = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in query.items()}
        idx = batch["idx"]

        start_time = time.time()
        ans = model.searchV2(args, query)
        pred = [str(l) for l in ans.cpu().tolist()]
        union_image = ividx.search_union(pred)
        time_list.append(time.time() - start_time)

        # if i >= 1000:
        #     break

        # if len(union_image) < args.max_topK:
        #     continue
        rank_score = {}

        if args.abl_setting == "ind_ret":
            for img_id, score in union_image:
                if img_id in rank_score:
                    rank_score[img_id] += 1
                else:
                    rank_score[img_id] = 1
        else:
            for img_id, score in union_image:
                if img_id in rank_score:
                    rank_score[img_id] += score
                else:
                    rank_score[img_id] = score
        # sorted_rank_score = sorted(rank_score.items(), key=lambda item: (-item[1], len(ividx.content[item[0]])))
        # TMP
        sorted_rank_score = sorted(rank_score.items(), key=lambda item: (-item[1], -len(ividx.content[item[0]])))
        rk = [item[0] for item in sorted_rank_score]
        # if len(rk) < args.max_topK:
        #     continue
        # noise level
        score_rk = [1.0 - imgid2level[imgid] for imgid in rk]

        rank_list.append(score_rk)
        rank_idx_list.append(rk)
        query_idx.append(idx[0].item())
        valid_use_query_i.append(i)

    # print(f"mean time:{np.mean(time_list)}")
    # exit(0)

    if args.abl_setting == "hard_way":
        # Save valid query i&idx
        with open(os.path.join("./data", args.dataset, f"valid_query_idx_{str(args.max_topK)}_T2I_noise{str(args.noise_rate)}.pkl"), "wb") as f:
            pickle.dump(valid_use_query_i, f)

        with open(os.path.join("./data", args.dataset, f"valid_query_idx_{str(args.max_topK)}_noise{str(args.noise_rate)}.pkl"), "wb") as f:
            pickle.dump(query_idx, f)

    # Val
    logger.info("Start evaluate.")
    def pad_results(lst, padding_value=0):
        max_len = max(len(sublist) for sublist in lst)
        padded_lst = [sublist + [padding_value] * (max_len - len(sublist)) for sublist in lst]
        return np.array(padded_lst)

    # rank_level = np.asarray(rank_list) # q, max_topK
    rank_level = pad_results(rank_list, padding_value=1.0)
    # rank = np.asarray(rank_idx_list)
    rank = pad_results(rank_idx_list, padding_value=-1)
    print(f"rank shape: {rank.shape}")

    ndcg_matrix = args.ndcg_matrix[query_idx, :]
    ndcg_matrix = ndcg_matrix[:, gallery_data.dataset.noise_img_idx_list]
    map_matrix = args.map_matrix[query_idx, :]
    map_matrix = map_matrix[:, gallery_data.dataset.noise_img_idx_list]

    def take_with_padding(indices_array, value_array):
        index = indices_array.copy()
        invalid_mask = (index == -1)  # 找到有效的索引位置
        index[index == -1] = 0
        output = np.take_along_axis(value_array, index, axis=1)
        output[invalid_mask] = 0.0
        return output

    # rank_rel = np.take_along_axis(ndcg_matrix, rank, axis=1)

    rank_rel = take_with_padding(rank, ndcg_matrix)

    topK_list = [1, 5, 10, 15, 50, args.max_topK]

    def compute_avg_score(ranks, test_matrix):
        query_size, db_size = ranks.shape
        return_score = []
        for i in range(query_size):
            score = 0
            i_ranks = ranks[i, :]
            cnt = 0
            for rk in i_ranks:
                if rk != -1:
                    score += test_matrix[i, rk]
                cnt += 1
            return_score.append(score / cnt)
        return np.average(return_score)

    def compute_avg_scoreV2(ranks):
        row_means = np.mean(ranks, axis=1)
        return np.mean(row_means)

    map_list = []
    ndcg_list = []
    acg_list = []
    ndcg_noise_list = []
    for k in tqdm(topK_list):
        rank_k = rank[:, :k]
        ndcg_list.append(compute_ndcg(rank_rel.T, k))
        acg_list.append(compute_avg_score(rank_k, ndcg_matrix))
        map_list.append(compute_mAP(map_matrix, rank_k.T))
        ndcg_noise_list.append(compute_ndcg(rank_level.T, k))

    logger.info(f"nDCG: {ndcg_list}")
    logger.info(f"acg: {acg_list}")
    logger.info(f"map: {map_list}")
    logger.info(f"noise nDCG: {ndcg_noise_list}")
    return np.mean(ndcg_list + acg_list)

@torch.no_grad()
def SIRMM_inference(
        args,
        model,
        query_data,
        gallery_data,
        logger,
        cfg):

    dict_tree = TreeNode()
    ividx = InvertedIndex()
    model.eval()

    # # construct id
    # # ps: suppose B = 1
    if os.path.isfile(os.path.join(args.model_path ,"y_list.pkl")):
        with open(os.path.join(args.model_path, "y_list.pkl"), "rb") as f:
            final_y_list_dic = pickle.load(f)
        final_y_list = final_y_list_dic["final_y_list"]
    else:
        final_y_list = []
        save_y_list = {}
        for i, batch in enumerate(tqdm(gallery_data)):
            img = batch["im"].float().to(args.device)
            idx = batch["idx"].long()
            y, enc_output = model(query=img, stage="tokenizer")
            y_list = y.int().tolist()
            final_y_list.append((int(idx[0].item()), y_list))
        save_y_list["final_y_list"] = final_y_list
        # save_y_list["gallery_feature"] = gallery_feature
        with open(os.path.join(args.model_path, "y_list.pkl"), "wb") as f:
            pickle.dump(save_y_list, f)

    for indices, y_list in final_y_list:
        dict_tree.insert_many(y_list)
        ividx_list = [f"{int(pair[0])},{int(pair[1])}" for pair in y_list]
        ividx.add_content(indices, ividx_list)

    ividx.display_index()

    # beam search
    ranks = []
    valid_query_idx = []
    valid_use_query_i = []
    for i, batch in enumerate(query_data):
        query = batch["caption"]
        query = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in query.items()}
        idx = batch["idx"]
        ans, score = model.module.search(args, query, beam_size=args.beam_size, dict_tree=dict_tree)
        q_idx = batch["idx"].long()
        rank_score, pred = rrank_origin(ans, score, ividx)
        print(f"query idx: {q_idx} score: {rank_score[:args.return_size]}")
        debug(batch, pred, cfg, ividx, args)

        if len(pred) >= args.return_size:
            ranks.append(pred[:args.return_size])
            valid_query_idx.append(int(idx[0].item()))
            valid_use_query_i.append(i)

    with open(os.path.join(args.meta_dir, f"valid_query_idx_{str(args.return_size)}_T2I.pkl"), "wb") as f:
        pickle.dump(valid_use_query_i, f)

    with open(os.path.join(args.meta_dir, f"valid_query_idx_{str(args.return_size)}.pkl"), "wb") as f:
        pickle.dump(valid_query_idx, f)

    ranks = np.asarray(ranks).T # beam_size * query
    print(f"ranks shape: {ranks.shape} query len:{len(query_data.dataset)} valid len: {len(valid_query_idx)}")
    # Test
    from evaluate import compute_mAP, compute_ndcg, compute_accuracy, compute_precision, compute_recall, compute_f1
    metric = {}
    sim_matrix = cfg["sim_matrix"]
    if args.dataset == "MultiScene-Clean":
        match_matrix = (sim_matrix > 0).astype(int)
    else:
        match_matrix = (sim_matrix > 0.5).astype(int)
    # valid correct
    metric["mAP100"] = compute_mAP(match_matrix[valid_query_idx, :], ranks)
    metric["nDCG"] = compute_ndcg(sim_matrix[valid_query_idx, :], ranks.shape[0], ranks)
    metric["accuracy"] = compute_accuracy(sim_matrix[valid_query_idx, :], ranks)
    precision_score_matrix = cfg["precision_score_matrix"]
    recall_score_matrix = cfg["recall_score_matrix"]
    f1_score_matrix = cfg["f1_score_matrix"]
    metric["precision"] = compute_precision(precision_score_matrix[valid_query_idx, :], ranks)
    metric["recall100"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks)
    metric["recall1"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:1, :])
    metric["recall5"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:5, :])
    metric["recall10"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:10, :])
    metric["recall30"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:30, :])
    metric["recall50"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:50, :])
    metric["f1"] = compute_f1(f1_score_matrix[valid_query_idx, :], ranks)

    logger.info('Origin mAP@{}:{}, nDCG@{}:{} Accuracy@{}:{} precision@{}:{} recall@{}:{} f1@{}:{}'.format(args.return_size, metric["mAP100"],
                                               args.return_size, metric["nDCG"],
                                               args.return_size, metric["accuracy"],
                                               args.return_size, metric["precision"],
                                               args.return_size, metric["recall100"],
                                               args.return_size, metric["f1"]))
    logger.info(
        'Origin recall@1/5/10/30/50:{},{},{},{},{}'.format(metric["recall1"],metric["recall5"],metric["recall10"],metric["recall30"],metric["recall50"]))

@torch.no_grad()
def SIRMM_inferenceV2(
        args,
        model,
        query_data,
        gallery_data,
        logger):

    ividx = InvertedIndex()
    model.eval()
    # Gallery
    logger.info("Compute Gallery.")
    id_list = []
    for i, batch in enumerate(tqdm(gallery_data)):
        img = batch["im"].to(args.device).float()
        idx = batch["idx"].int()
        identifier, embed = model(query=img, stage="tokenizer")  # (N, 2)
        id_list.append((int(idx[0].item()), identifier.tolist()))

    for idx, identifier in id_list:
        content_list = [(str(int(pair[0])), float(pair[1])) for pair in identifier]
        ividx.add_contentV2(idx, content_list)

    ividx.display_index()

    # beam search
    ranks = []
    valid_query_idx = []
    valid_use_query_i = []
    for i, batch in enumerate(query_data):
        query = batch["caption"]
        query = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in query.items()}
        idx = batch["idx"]
        ans, score = model.module.search(args, query, beam_size=args.beam_size, dict_tree=dict_tree)
        q_idx = batch["idx"].long()
        rank_score, pred = rrank_origin(ans, score, ividx)
        print(f"query idx: {q_idx} score: {rank_score[:args.return_size]}")
        debug(batch, pred, cfg, ividx, args)

        if len(pred) >= args.return_size:
            ranks.append(pred[:args.return_size])
            valid_query_idx.append(int(idx[0].item()))
            valid_use_query_i.append(i)

    with open(os.path.join(args.meta_dir, f"valid_query_idx_{str(args.return_size)}_T2I.pkl"), "wb") as f:
        pickle.dump(valid_use_query_i, f)

    with open(os.path.join(args.meta_dir, f"valid_query_idx_{str(args.return_size)}.pkl"), "wb") as f:
        pickle.dump(valid_query_idx, f)

    ranks = np.asarray(ranks).T # beam_size * query
    print(f"ranks shape: {ranks.shape} query len:{len(query_data.dataset)} valid len: {len(valid_query_idx)}")
    # Test
    from evaluate import compute_mAP, compute_ndcg, compute_accuracy, compute_precision, compute_recall, compute_f1
    metric = {}
    sim_matrix = cfg["sim_matrix"]
    if args.dataset == "MultiScene-Clean":
        match_matrix = (sim_matrix > 0).astype(int)
    else:
        match_matrix = (sim_matrix > 0.5).astype(int)
    # valid correct
    metric["mAP100"] = compute_mAP(match_matrix[valid_query_idx, :], ranks)
    metric["nDCG"] = compute_ndcg(sim_matrix[valid_query_idx, :], ranks.shape[0], ranks)
    metric["accuracy"] = compute_accuracy(sim_matrix[valid_query_idx, :], ranks)
    precision_score_matrix = cfg["precision_score_matrix"]
    recall_score_matrix = cfg["recall_score_matrix"]
    f1_score_matrix = cfg["f1_score_matrix"]
    metric["precision"] = compute_precision(precision_score_matrix[valid_query_idx, :], ranks)
    metric["recall100"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks)
    metric["recall1"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:1, :])
    metric["recall5"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:5, :])
    metric["recall10"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:10, :])
    metric["recall30"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:30, :])
    metric["recall50"] = compute_recall(recall_score_matrix[valid_query_idx, :], ranks[:50, :])
    metric["f1"] = compute_f1(f1_score_matrix[valid_query_idx, :], ranks)

    logger.info('Origin mAP@{}:{}, nDCG@{}:{} Accuracy@{}:{} precision@{}:{} recall@{}:{} f1@{}:{}'.format(args.return_size, metric["mAP100"],
                                               args.return_size, metric["nDCG"],
                                               args.return_size, metric["accuracy"],
                                               args.return_size, metric["precision"],
                                               args.return_size, metric["recall100"],
                                               args.return_size, metric["f1"]))
    logger.info(
        'Origin recall@1/5/10/30/50:{},{},{},{},{}'.format(metric["recall1"],metric["recall5"],metric["recall10"],metric["recall30"],metric["recall50"]))