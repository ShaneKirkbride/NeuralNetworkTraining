
using TorchSharp;

namespace NeuralNetworkTraining.ViewModel;

using System.IO;
using System.Collections.ObjectModel;
using System.Windows;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.ML;
using Microsoft.ML.TensorFlow;
using NeuralNetworkTraining.Models;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using Tensorflow.Keras.Losses;
using Tensorflow;
using System;
using static TorchSharp.torch.optim.lr_scheduler;

public partial class MainViewModel : ObservableObject
{
    public ObservableCollection<long> OutputOptions { get; } = new() { 3, 6, 12 };
    public ObservableCollection<string> OperationModes { get; } = new() { "Generate Data", "Train Model", "Run Inference" };
    public ObservableCollection<PredictionResult> Predictions { get; } = new();

    [ObservableProperty]
    private long selectedOutput;

    [ObservableProperty]
    private string selectedOperationMode;

    [ObservableProperty]
    private string inputFilePath;

    [ObservableProperty]
    private string trainingDataPath;

    [ObservableProperty]
    private string generatedDataPath;

    [ObservableProperty]
    private string modelDirectory = "./models";

    private Sequential torchModel;
    private string modelPath;
    public ObservableCollection<string> Logs { get; } = new();

    public MainViewModel()
    {
        ExecuteOperationCommand = new RelayCommand(ExecuteOperation);
    }

    public IRelayCommand ExecuteOperationCommand { get; }

    private void ExecuteOperation()
    {
        switch (SelectedOperationMode)
        {
            case "Generate Data":
                if (!string.IsNullOrWhiteSpace(GeneratedDataPath))
                {
                    GenerateTrainingData(GeneratedDataPath);
                }
                else
                {
                    MessageBox.Show("Please specify a valid file path for generated data.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
                break;
            case "Train Model":
                TrainTorchModel();
                break;
            case "Run Inference":
                LoadAndPredict();
                break;
            default:
                MessageBox.Show("Please select a valid operation mode.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                break;
        }
    }

    public void GenerateTrainingData(string outputPathTraining, int samples = 100)
    {
        var random = new Random();
        var trainingData = new List<string>();
        var validationData = new List<string>();
        string outputPathValidation = outputPathTraining + "_validation.csv";
        outputPathTraining = outputPathTraining + ".csv";

        var pristineSineWave = GeneratePristineSineWave(1024, SelectedOutput / 2);
        var peakIndices = IdentifyPeaks(pristineSineWave);

        GenerateDataSet(trainingData, peakIndices, samples, true);
        GenerateDataSet(validationData, peakIndices, samples, true);

        File.WriteAllLines(outputPathTraining, trainingData);
        Logs.Add($"Training data generated at {outputPathTraining}");
        MessageBox.Show($"Training data generated at {outputPathTraining}", "Info", MessageBoxButton.OK, MessageBoxImage.Information);

        File.WriteAllLines(outputPathValidation, validationData);
        Logs.Add($"Validation data generated at {outputPathValidation}");
        MessageBox.Show($"Validation data generated at {outputPathValidation}", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
    }

    private float[] GeneratePristineSineWave(int length, long cycles)
    {
        var sineWave = new float[length];
        for (int j = 0; j < length; j++)
        {
            float x = (float)j / length;
            sineWave[j] = Math.Abs((float)Math.Sin(2 * Math.PI * cycles * x));
        }
        return sineWave;
    }

    private List<int> IdentifyPeaks(float[] signal)
    {
        // Identify peaks (bumps) in the signal
        var peaks = new List<int>();
        for (int j = 1; j < signal.Length - 1; j++)
        {
            if (signal[j] > signal[j - 1] && signal[j] > signal[j + 1])
            {
                peaks.Add(j);
            }
        }
        return peaks;
    }

    private void GenerateDataSet(List<string> dataset, List<int> peakIndices, int samples, bool addNoise)
    {
        var random = new Random();
        for (int i = 0; i < samples; i++)
        {
            var features = new float[1024];
            for (int j = 0; j < features.Length; j++)
            {
                float x = (float)j / 1024;
                float noise = addNoise ? (float)(random.NextDouble() * 1.001 - 0.999) : 1;
                features[j] = Math.Abs((float)(noise*Math.Sin(2 * Math.PI * (SelectedOutput / 2) * x)));
            }

            var labels = new List<float>();
            foreach (var peak in peakIndices)
            {
                float maxValue = features[peak];
                int leftHalf = -1;
                for (int k = peak - 1; k >= 0; k--)
                {
                    if (features[k] <= maxValue / 2)
                    {
                        leftHalf = k;
                        break;
                    }
                }

                int rightHalf = -1;
                for (int k = peak + 1; k < features.Length; k++)
                {
                    if (features[k] <= maxValue / 2)
                    {
                        rightHalf = k;
                        break;
                    }
                }
                labels.Add(peak);
                labels.Add(maxValue);
                labels.Add(leftHalf);
                labels.Add(features[leftHalf]);
                labels.Add(rightHalf);
                labels.Add(features[rightHalf]);
            }
            dataset.Add(string.Join(",", features) + "," + string.Join(",", labels));
        }
    }

    private void TrainTorchModel()
    {
        if (string.IsNullOrWhiteSpace(TrainingDataPath) || SelectedOutput <= 0)
        {
            MessageBox.Show("Please select valid training data and output size.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        var trainingData = InputDataLoader.LoadInputData(TrainingDataPath);
        var inputs = ConvertTo2DTensor(trainingData.Select(d => d.Features).ToArray());
        var labels = ConvertTo2DTensor(trainingData.Select(d => d.Labels).ToArray());
        this.SelectedOutput = labels.size(1);

        // Define the model using nn.Sequential
        torchModel = nn.Sequential(
            ("lin1", nn.Linear(1024, 512)),
            ("relu1", nn.LeakyReLU(0.01)),  // ReLU replaced with LeakyReLU
            ("lin2", nn.Linear(512, 512)),
            ("relu2", nn.LeakyReLU(0.01)),
            ("lin3", nn.Linear(512, 256)),
            ("relu3", nn.LeakyReLU(0.01)),
            ("lin4", nn.Linear(256, 256)),
            ("relu4", nn.LeakyReLU(0.01)),
            ("lin5", nn.Linear(256, 128)),
            ("relu5", nn.LeakyReLU(0.01)),
            ("lin6", nn.Linear(128, 128)),
            ("relu6", nn.LeakyReLU(0.01)),
            ("lin7", nn.Linear(128, 64)),
            ("relu7", nn.LeakyReLU(0.01)),
            ("lin8", nn.Linear(64, 64)),
            ("relu8", nn.LeakyReLU(0.01)),
            ("lin9", nn.Linear(64, 32)),
            ("relu9", nn.LeakyReLU(0.01)),
            ("lin10", nn.Linear(32, SelectedOutput))
        );

        foreach (var layer in torchModel.children())
        {
            if (layer is Linear linearLayer)
            {
                torch.nn.init.kaiming_uniform_(linearLayer.weight);
            }
        }

        var learningRate = 0.001f;
        var optimizer = torch.optim.Adam(torchModel.parameters(), learningRate);
        //var scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size: 100, gamma: 0.1f); // Reduce LR every 1000 epochs

        var lossFunction = nn.MSELoss();

        // Training loop
        for (int epoch = 0; epoch < 20000; epoch++)
        {
            torchModel.train();
            var predictions = torchModel.forward(inputs);
            var loss = lossFunction.forward(predictions, labels);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            //scheduler.step(); // Adjust learning rate

            var log = $"Epoch {epoch + 1}, Loss: {loss.item<float>()}";
            //var log = $"Epoch {epoch + 1}, Loss: {loss.item<float>()}, LR: {scheduler.get_last_lr().First()}";
            Logs.Add(log);

            if (loss.item<float>() <= 2e-5)
            {
                Logs.Add($"Loss tolerance reached: {loss.item<float>()}");
                break;
            }
        }

        SaveTorchModel();
    }

    private void SaveTorchModel()
    {
        this.modelPath = Path.Combine(ModelDirectory, $"torch_model_{SelectedOutput}.bin");
        this.torchModel.save(this.modelPath);
        Logs.Add($"TorchSharp model trained and saved to {modelPath}.");
        MessageBox.Show($"TorchSharp model trained and saved to {this.modelPath}.", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
    }

    private void LoadAndPredict()
    {
        if (string.IsNullOrWhiteSpace(InputFilePath) || SelectedOutput <= 0)
        {
            MessageBox.Show("Please select a valid input file and output size.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        this.modelPath = Path.Combine(ModelDirectory, $"torch_model_{SelectedOutput}.bin");

        if (!File.Exists(this.modelPath))
        {
            MessageBox.Show($"TorchSharp model file {this.modelPath} not found.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        torchModel = nn.Sequential(
            ("lin1", nn.Linear(1024, 512)),
            ("relu1", nn.LeakyReLU(0.01)),  // ReLU replaced with LeakyReLU
            ("lin2", nn.Linear(512, 512)),
            ("relu2", nn.LeakyReLU(0.01)),
            ("lin3", nn.Linear(512, 256)),
            ("relu3", nn.LeakyReLU(0.01)),
            ("lin4", nn.Linear(256, 256)),
            ("relu4", nn.LeakyReLU(0.01)),
            ("lin5", nn.Linear(256, 128)),
            ("relu5", nn.LeakyReLU(0.01)),
            ("lin6", nn.Linear(128, 128)),
            ("relu6", nn.LeakyReLU(0.01)),
            ("lin7", nn.Linear(128, 64)),
            ("relu7", nn.LeakyReLU(0.01)),
            ("lin8", nn.Linear(64, 64)),
            ("relu8", nn.LeakyReLU(0.01)),
            ("lin9", nn.Linear(64, 32)),
            ("relu9", nn.LeakyReLU(0.01)),
            ("lin10", nn.Linear(32, SelectedOutput))
        );


        torchModel.load(this.modelPath, false);

        var inputData = InputDataLoader.LoadInputData(InputFilePath);
        var inputs = ConvertTo2DTensor(inputData.Select(d => d.Features).ToArray());

        torchModel.eval();
        var predictions = torchModel.forward(inputs);

        Predictions.Clear();

        for (int i = 0; i < inputData.Length; i++)
        {
            for (int labelIndex = 0; labelIndex < inputData[i].Labels.Length; labelIndex++)
            {
                Predictions.Add(new PredictionResult
                {
                    Features = inputData[i].Features,
                    PredictedValue = predictions[i][labelIndex].item<float>(), // Access the correct prediction for the label
                    Accuracy = AccuracyCalculator.CalculateAccuracy(predictions[i][labelIndex].item<float>(), inputData[i].Labels[labelIndex])
                });
            }
        }
        Logs.Add("Inference completed and predictions updated.");
    }

    private torch.Tensor ConvertTo2DTensor(float[][] jaggedArray)
    {
        int rows = jaggedArray.Length;
        int cols = jaggedArray[0].Length;
        var flatArray = new float[rows * cols];
        for (int i = 0; i < rows; i++)
        {
            Array.Copy(jaggedArray[i], 0, flatArray, i * cols, cols);
        }
        return torch.tensor(flatArray, new long[] { rows, cols }, torch.float32);
    }
}

public class PredictionResult
{
    public float[] Features { get; set; }
    public float PredictedValue { get; set; }
    public float Accuracy { get; set; }
}