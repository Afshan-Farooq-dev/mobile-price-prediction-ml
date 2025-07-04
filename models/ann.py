import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import numpy as np

class ANNModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.history = None
    
    def build_model(self, input_dim, num_classes):
        """Build the neural network architecture"""
        self.model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train):
        """Train the neural network"""
        try:
            # Build model if not built
            if self.model is None:
                num_classes = len(np.unique(y_train))
                self.build_model(X_train.shape[1], num_classes)
            
            # Train the model
            self.history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0,  # Silent training
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training ANN: {e}")
            return False
    
    def predict(self, X_test):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X_test, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X_test, verbose=0)
    
    def get_feature_importance(self, feature_names):
        """Neural networks don't provide direct feature importance"""
        return None
    
    def get_training_history(self):
        """Get training history for plotting"""
        if self.history is None:
            return None
        return {
            'loss': self.history.history['loss'],
            'accuracy': self.history.history['accuracy'],
            'val_loss': self.history.history['val_loss'],
            'val_accuracy': self.history.history['val_accuracy']
        }
    
    def get_model_info(self):
        """Get model information"""
        return {
            'name': 'Artificial Neural Network (ANN)',
            'type': 'Deep Learning Classification',
            'description': 'Multi-layer neural network with backpropagation learning',
            'pros': ['Can learn complex patterns', 'Flexible architecture', 'Good for large datasets'],
            'cons': ['Black box model', 'Requires more data', 'Computationally expensive', 'Many hyperparameters']
        }



#         Important terms in ANN
# Weights: Ye numbers hote hain jo neurons ke connections ko strong ya weak karte hain.

# Bias: Ye ek extra number hota hai jo neuron ki output ko adjust karta hai.

# Activation Function: Ye decide karta hai ke neuron ka output kya hoga (jaise ReLU, Sigmoid, Tanh).

# Forward Propagation: Data ka network mein aage badhna.

# Loss Function: Ye batata hai ke model ki prediction kitni galat hai.

# Backpropagation: Training ka process jisme model apni galti ko samajhta hai aur weights ko update karta hai taake agle predictions behtar hoon.

# Epoch: Training ke complete cycle ko kehte hain jisme sara training data model se guzarta hai.

