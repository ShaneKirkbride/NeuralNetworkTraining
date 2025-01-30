using Microsoft.ML.Data;
using System.IO;

namespace NeuralNetworkTraining.Models
{
    /// <summary>
    /// Input data schema.
    /// </summary>
    public class InputData
    {
        public float[] Features { get; set; } // 1024 inputs
        public float[] Labels { get; set; }   // Dynamic number of labels per sample
    }

    /// <summary>
    /// Output data schema.
    /// </summary>
    public class OutputData
    {
        [ColumnName("Score")]
        public float[] PredictedLabels { get; set; }
    }

    /// <summary>
    /// Utility class for loading input data.
    /// </summary>
    public static class InputDataLoader
    {
        public static InputData[] LoadInputData(string filePath)
        {
            var data = new List<InputData>();
            using (var reader = new StreamReader(filePath))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var values = line.Split(',');
                    var featureLength = 1024; // First 1024 are features
                    var features = values[..featureLength].Select(float.Parse).ToArray();
                    var labels = values[featureLength..].Select(float.Parse).ToArray();

                    data.Add(new InputData
                    {
                        Features = features,
                        Labels = labels
                    });
                }
            }

            return data.ToArray();
        }
    }


    /// <summary>
    /// Utility class for calculating accuracy.
    /// </summary>
    public static class AccuracyCalculator
    {
        public static float CalculateAccuracy(float predicted, float actual)
        {
            return 1.0f - Math.Abs(predicted - actual) / (Math.Abs(actual) + 1e-6f);
        }
    }
}
