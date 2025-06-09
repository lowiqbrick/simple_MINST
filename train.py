# preexisting modules
import torch
import torch.optim as optim
from tqdm import tqdm
# custom modules
from model import MNIST_network
from loss_function import MNIST_loss
import configurations
import utility
import time
import warnings
warnings.filterwarnings("ignore")


def train(data_loader, network, optimizer, loss, scaler):
    data_loader_tqdm = tqdm(data_loader)
    loss_list = []
    for (images, results) in data_loader_tqdm:
        # print(images[0])
        images = images.to(configurations.DEVICE)
        results = results.to(configurations.DEVICE)

        with torch.cuda.amp.autocast():
            output = network(images)
            loss_iteration = loss(output, results)

        loss_list.append(loss_iteration.item())
        optimizer.zero_grad()
        scaler.scale(loss_iteration).backward()
        scaler.step(optimizer)
        scaler.update()
        av_loss = sum(loss_list)/len(loss_list)
        data_loader_tqdm.set_postfix(av_loss=av_loss)


def main():
    network = MNIST_network().to(configurations.DEVICE)
    optimizer = optim.Adam(network.parameters(),
                           configurations.LEARNING_RATE,
                           weight_decay=configurations.WEIGHT_DECAY)
    loss = MNIST_loss()
    train_loader, test_loader = utility.get_loaders(
        configurations.PATH_TO_FOLDER)
    MNIST_scaler = torch.cuda.amp.GradScaler()

    if configurations.LOAD:
        utility.load_model(configurations.BEST_FILE,
                           network, optimizer,
                           configurations.LEARNING_RATE)

    # create input and its optimizer
    # repeat converts [1, 1, 28, 28] to [2, 1, 28, 28]
    input = torch.rand([1, 1, 28, 28], requires_grad=True).repeat(2, 1, 1, 1)
    input = input.detach()
    print(input.shape)

    # setting test_accuracy
    last_test_accuracy = configurations.BEST_SO_FAR
    print("learning rate: " + str(configurations.LEARNING_RATE))
    print("batch size: " + str(configurations.BATCH_SIZE))
    print("number of workers: " + str(configurations.NUM_WORKERS))
    print("load model: " + str(configurations.LOAD) + " (" +
          str(configurations.BEST_FILE) + ")")
    print("save model: " + str(configurations.SAVE))
    for epoch in range(1, configurations.NUM_EPOCHS):
        network.train()
        # """
        train(train_loader, network, optimizer, loss, MNIST_scaler)  # """
        network.eval()
        if configurations.SAVE:
            utility.save_model(network, optimizer,
                               configurations.CHECK_FILE_NAME)
            """
            utility.save_model(network, optimizer,
                            str(int(time.time())) +
                            configurations.CHECK_FILE_NAME)  # """
        if (epoch % 5 == 0):
            # calculate accuracy
            train_accuracy = utility.calculate_accuracy(network,
                                                        train_loader)
            print("current train accuracy at " + str(round(train_accuracy,
                                                           2))
                  + "%")
            test_accuracy = utility.calculate_accuracy(network,
                                                       test_loader)
            print("current test accuracy at " + str(round(test_accuracy,
                                                          2))
                  + "%")

            # save model
            """
            if configurations.SAVE:
                utility.save_model(
                        network,
                        optimizer,
                        str(round(train_accuracy, 1)) + " " +
                        str(round(test_accuracy, 1)) + " " +
                        configurations.CHECK_FILE_NAME)  # """
            # save best model
            if test_accuracy > last_test_accuracy and\
                    test_accuracy > configurations.BEST_SO_FAR:
                utility.save_model(network, optimizer,
                                   str(round(test_accuracy, 2)) + " top " +
                                   configurations.CHECK_FILE_NAME)
            # update last_test_accuracy
            last_test_accuracy = test_accuracy


if __name__ == "__main__":
    main()