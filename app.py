import streamlit as st
from backend import *
import pandas as pd
import matplotlib.pyplot as plot

from bokeh.plotting import figure
def train(optimizername,namestr):
  # Create dataset
  X, y = spiral_data(samples=100, classes=3)
  acc =[]

  # Create Dense layer with 2 input features and 64 output values
  dense1 = Layer_Dense(2, 64)

  # Create ReLU activation (to be used with Dense layer):
  activation1 = Activation_ReLU()

  # Create second Dense layer with 64 input features (as we take output
  # of previous layer here) and 3 output values (output values)
  dense2 = Layer_Dense(64, 3)
  # Create Softmax classifier's combined loss and activation
  loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

  # Create optimizer
  optimizer = optimizername
  with st.sidebar:
    st.header(namestr)
    # Train in loop
    for epoch in range(10001):

        # Perform a forward pass of our training data through this layer
        dense1.forward(X)

        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)

        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs
        dense2.forward(activation1.output)

        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        loss = loss_activation.forward(dense2.output, y)

        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions==y)

        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f}, ' +
                    f'lr: {optimizer.current_learning_rate}')
            acc.append(accuracy)
        
        if  epoch== 10000:
            st.markdown(f'Number of Epochs: {epoch} ')
            st.markdown(f'Accuracy: {accuracy:.3f}, '+f'Loss: {loss:.3f} ')
            st.markdown(f'Learning Rate: {optimizer.current_learning_rate}')

        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
  return acc

def main():
    st.title("Neural Network Optimizers Comparison with Quantum-Generated Noise")
    st.header("Implementing Differential Privacy in Federated Learning")

    st.write("""

    In this application, we explore the impact of incorporating quantum-generated noise on the performance of various optimization algorithms used in training neural networks. The goal is to preserve differential privacy, a crucial aspect of federated learning, while minimizing the sacrifice in model accuracy.

    We compare the following optimizers:

    - **SGD (Stochastic Gradient Descent)**
    - **SGD with Momentum**
    - **SGD with Decay**
    - **Adagrad**
    - **RMSprop**
    - **Adam**

    By leveraging quantum algorithms, we introduce noise to the training process in a controlled and efficient manner, allowing for enhanced privacy protection while maintaining model performance.

    
    We observe how the optimizers behave in the presence of quantum-generated noise, gaining valuable insights into the trade-off between privacy and accuracy in federated learning scenarios.
    """)
   # Create a form for user input
    with st.form("optimizer_params"):
       st.write("## Optimizer Parameters")

       # SGD
       sgd_lr = st.number_input("SGD Learning Rate", value=1.0, step=0.1)
       sgd_decay = st.number_input("SGD Decay", value=0.0, step=0.001)
       sgd_momentum = st.number_input("SGD Momentum", value=0.0, min_value=0.0, max_value=1.0, step=0.1)

    #    # SGD with Momentum
    #    sgd_mom_lr = st.number_input("SGD with Momentum Learning Rate", value=0.85, step=0.01)
    #    sgd_mom_decay = st.number_input("SGD with Momentum Decay", value=0.001, step=0.0001)
    #    sgd_mom_momentum = st.number_input("SGD with Momentum Momentum", value=0.9, min_value=0.0, max_value=1.0, step=0.1)

    #    # SGD with Decay
    #    sgd_decay_lr = st.number_input("SGD with Decay Learning Rate", value=0.85, step=0.01)
    #    sgd_decay_decay = st.number_input("SGD with Decay Decay", value=0.001, step=0.0001)

       submitted = st.form_submit_button("Start Training Models")

    if submitted:
        SGD_optimizer_with_momentum = Optimizer_SGD(learning_rate=0.85,decay=1e-3,momentum=0.9)
        SGD_optimizer_lr_decay = Optimizer_SGD(learning_rate=0.85,decay=1e-3,momentum=0)
        SGD_optimizer = Optimizer_SGD(learning_rate=sgd_lr,decay=sgd_decay,momentum=sgd_momentum)
        adagrad_optimizer = Optimizer_Adagrad()
        rmsprop_optimizer = Optimizer_RMSprop()
        adam_optimizer = Optimizer_Adam()

        # Train with optimizers
        SGD_with_decay_momentum_acc=train(optimizername=SGD_optimizer_with_momentum,namestr="SGD with Decay and Momentum")
        SGD_with_decay_acc = train(optimizername=SGD_optimizer_lr_decay,namestr="SGD with LR Decay")
        SGD_with_const_lr_acc = train(optimizername=SGD_optimizer, namestr="SGD with constant LR")
        adagrad_acc = train(optimizername=adagrad_optimizer,namestr="ADA Grad")
        rmsprop_acc = train(optimizername=rmsprop_optimizer,namestr="RMSprop")
        adam_acc = train(optimizername=adam_optimizer,namestr="ADAM")

        epoch = [x for x in range(0, 10001, 100)]
        acc_dict = {"Epoch": epoch,
                "Adam": adam_acc,
                "RMSprop": rmsprop_acc,
                "Adagrad":adagrad_acc,
                #"SGD":sgd_acc}
                "SGD_with_decay_momentum":SGD_with_decay_momentum_acc,
                "SGD_with_decay": SGD_with_decay_acc,
                "SGD_with_const_lr": SGD_with_const_lr_acc}
        df = pd.DataFrame.from_dict(acc_dict)

        # Plot the figure
        # fig, ax = plt.subplots(figsize=(14, 7))
        # ax.plot(df["Epoch"], df[df.columns[1:]])
        # ax.legend(df.columns[1:], loc="best")
        # ax.set_xlabel("Epochs")
        # ax.set_ylabel("Accuracy")
        # ax.set_title('Comparison of Optimizers', fontsize=14, fontweight='bold')
        # st.pyplot(fig)
        p = figure(
        title='Comparision of Algorithms',
        x_axis_label='Number of Epochs',
        y_axis_label='Accuracy')

        p.line(df["Epoch"],df["Adam"], legend_label='ADAM',line_color='green', line_width=2)
        p.line(df["Epoch"],df["RMSprop"], legend_label='RMS Prop',line_color='red', line_width=2)
        p.line(df["Epoch"],df["Adagrad"], legend_label='ADA Grad',line_color='black', line_width=2)
        p.line(df["Epoch"],df["SGD_with_decay_momentum"], legend_label='SGD with Momentum',line_color='orange', line_width=2)
        p.line(df["Epoch"],df["SGD_with_decay"], legend_label='SGD with Decay',line_color='blue', line_width=2)
        p.line(df["Epoch"],df["SGD_with_const_lr"], legend_label='SGD with Constant Learning Rate',line_color='brown',line_width=2)
    
        st.bokeh_chart(p, use_container_width=True)

if __name__ == "__main__":
    main()