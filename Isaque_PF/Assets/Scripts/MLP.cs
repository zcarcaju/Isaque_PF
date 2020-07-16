using System.IO;
using System.Globalization;
using UnityEngine;

public class MLP : MonoBehaviour
{
    const int numInputs = 2;
    const int neuronsHidden = 5;
    const int neuronsOut = 4;
    const int numGens = 30000;
    const int costRequisite = 170;
    const double cost = 0.7;
    const double learningRate = 0.15;

    double[,] wInpHid = new double[neuronsHidden, numInputs];
    double[,] wHidOut = new double[neuronsOut, neuronsHidden];

    string fileTrain;
    string[] trainFileInputs;
    double[,] trainInputsFile;
    double[] expectedGestures;
    
    void InitializeWeights()
    {
        for (int i = 0; i < wInpHid.GetLength(0); ++i)
        {
            for (int j = 0; j < wInpHid.GetLength(1); ++j)
            {
                wInpHid[i, j] = Random.value;
            }
        }

        for (int i = 0; i < wHidOut.GetLength(0); ++i)
        {
            for (int j = 0; j < wHidOut.GetLength(1); ++j)
            {
                wHidOut[i, j] = Random.value;
            }
        }
    }

    void ReadTrainFile()
    {
        fileTrain = Application.dataPath + "\\../Dataset.txt";

        trainFileInputs = File.ReadAllLines(fileTrain);
        trainInputsFile = DoubleInputs(trainFileInputs);
        expectedGestures = ExpectedResult(trainFileInputs);
    }

    double[,] DoubleInputs(string[] fileRead)
    {
        NumberFormatInfo separator = (NumberFormatInfo)CultureInfo.CurrentCulture.NumberFormat.Clone();
        separator.NumberDecimalSeparator = ".";

        int h = fileRead[0].Split(',').Length - 1;
        double[,] matrixReturn = new double[fileRead.Length, h];

        for (int i = 0; i < fileRead.Length; ++i)
        {
            for (int j = 0; j < h; ++j)
            {
                matrixReturn[i, j] = double.Parse(fileRead[i].Split(',')[j], separator);
            }
        }

        return matrixReturn;
    }

    double[] ExpectedResult(string[] gestures)
    {
        NumberFormatInfo separator = (NumberFormatInfo)CultureInfo.CurrentCulture.NumberFormat.Clone();
        separator.NumberDecimalSeparator = ".";

        int h = gestures[0].Split(',').Length - 1; 
        double[] arrayReturn = new double[gestures.Length];

        for (int i = 0; i < gestures.Length; ++i)
        {
            arrayReturn[i] = double.Parse(gestures[i].Split(',')[h], separator);
        }

        return arrayReturn;
    }

    double OutputPerceptron(double[] pInputs, double[] pWeights)
    {
        double sum = 0.0;

        for (int i = 0; i < pInputs.Length; ++i)
        {
            sum += pInputs[i] * pWeights[i];
        }

        double output = 1.0 / (1.0 + System.Math.Exp(-sum)); ; //sigmoid

        return output;
    }

    double[] MatrixLineToArray(double[,] matrix, int line)
    {
        double[] lineReturn = new double[matrix.GetLength(1)];

        for (int i = 0; i < matrix.GetLength(1); ++i)
        {
            lineReturn[i] = matrix[line, i];
        }

        return lineReturn;
    }

    double SquaredError(double[] errors)
    {
        double squaredError = 0.0;

        foreach (double error in errors)
        {
            squaredError += error * error;
        }

        return squaredError;
    }

    void SaveTrainedWeights(double[,] trainedWeights, string networkLayer)
    {
        string path = "";

        switch (networkLayer)
        {
            case "Hidden":
                path = Application.dataPath + "\\..\\TrainedWeightsHidden.txt";
                break;
            case "Output":
                path = Application.dataPath + "\\..\\TrainedWeightsOutput.txt";
                break;
        }

        StreamWriter weightsFile = new StreamWriter(path);

        for (int i = 0; i < trainedWeights.GetLength(0); ++i)
        {
            string line = "";
            for (int j = 0; j < trainedWeights.GetLength(1); ++j)
            {
                if (j != trainedWeights.GetLength(1) - 1)
                {
                    line += trainedWeights[i, j].ToString().Replace(',', '.') + ",";
                }
                else
                {
                    line += trainedWeights[i, j].ToString().Replace(',', '.');
                }
            }
            weightsFile.WriteLine(line);
        }

        weightsFile.Close();
    }

    public void TrainNN()
    {
        InitializeWeights();
        ReadTrainFile();

        int currentGen = 0;

        for (int gen = 0; gen < numGens; ++gen)
        {
            int numCosts = 0;
            for (int i = 0; i < trainInputsFile.GetLength(0); ++i)
            {
                double[] resultHidden = new double[neuronsHidden];
                double[] resultOutput = new double[neuronsOut];
                double[] outputErrors = new double[neuronsOut];
                double[] derivativesOutput = new double[neuronsOut];
                double[] derivativesHidden = new double[neuronsHidden];
                double sumOutputErrors = 0.0;

                for (int j = 0; j < neuronsHidden; ++j)
                {
                    resultHidden[j] = OutputPerceptron(MatrixLineToArray(trainInputsFile, i), MatrixLineToArray(wInpHid, j));
                }
                for (int j = 0; j < neuronsOut; ++j)
                {
                    resultOutput[j] = OutputPerceptron(resultHidden, MatrixLineToArray(wHidOut, j));
                    outputErrors[j] = expectedGestures[i] - resultOutput[j];
                    derivativesOutput[j] = resultOutput[j] * (1.0 - resultOutput[j]);
                    sumOutputErrors += outputErrors[j] * derivativesOutput[j];
                }

                if (SquaredError(outputErrors) <= cost)
                {
                    ++numCosts;
                }

                for (int j = 0; j < neuronsOut; ++j)
                {
                    double error = outputErrors[j] * derivativesOutput[j];
                    for (int a = 0; a < wHidOut.GetLength(1); ++a)
                    {
                        wHidOut[j, a] += learningRate * error * resultHidden[a];
                    }
                }

                for (int j = 0; j < neuronsHidden; ++j)
                {
                    derivativesHidden[j] = resultHidden[j] * (1.0 - resultHidden[j]);
                    double error = derivativesHidden[j] * sumOutputErrors;
                    for (int a = 0; a < wInpHid.GetLength(1); ++a)
                    {
                        wInpHid[j, a] += learningRate * error * trainInputsFile[i, a];
                    }
                }
            }
            if (numCosts > costRequisite)
            {
                Debug.LogWarning("Treinamento concluído pelo custo.");
                break;
            }
            else if (currentGen >= numGens)
            {
                Debug.Log("Treinando novamente");
                TrainNN();
                break;
            }
        }
        Debug.LogWarning("Treinamento concluído!");
        SaveTrainedWeights(wInpHid, "Hidden");
        SaveTrainedWeights(wHidOut, "Output");
    }

    public void ValidateGesture(double distMagnitude, double numOfTiles)
    {
        string wInpHidFilePath = Application.dataPath + "\\..\\TrainedWeightsHidden.txt";
        string wHidOutFilePath = Application.dataPath + "\\..\\TrainedWeightsOutput.txt";

        string[] fileLines = File.ReadAllLines(wInpHidFilePath);
        double[,] trainedWInpHid = ReadTrainedWeightsFile(fileLines);

        fileLines = File.ReadAllLines(wHidOutFilePath);
        double[,] trainedWHidOut = ReadTrainedWeightsFile(fileLines);

        double[] inputs = new double[2] { distMagnitude, numOfTiles };
        double[] resultHidden = new double[neuronsHidden];
        double[] resultOutput = new double[neuronsOut];

        for (int i = 0; i < neuronsHidden; ++i)
        {
            resultHidden[i] = OutputPerceptron(inputs, MatrixLineToArray(trainedWInpHid, i));
        }
        for (int i = 0; i < neuronsOut; ++i)
        {
            resultOutput[i] = OutputPerceptron(resultHidden, MatrixLineToArray(trainedWHidOut, i));
        }

        Debug.Log(ValidatedOutput(resultOutput[0]));
    }

    double[,] ReadTrainedWeightsFile(string[] fileLines)
    {
        NumberFormatInfo separator = (NumberFormatInfo)CultureInfo.CurrentCulture.NumberFormat.Clone();
        separator.NumberDecimalSeparator = ".";

        double[,] matrixReturn = new double[fileLines.Length, fileLines[0].Split(',').Length];
        
        for (int i = 0; i < matrixReturn.GetLength(0); ++i)
        {
            for (int j = 0; j < matrixReturn.GetLength(1); ++j)
            {
                matrixReturn[i, j] = double.Parse(fileLines[i].Split(',')[j], separator);
            }
        }

        return matrixReturn;
    }

    string ValidatedOutput(double value)
    {
        Debug.Log("Value is: " + value);
        if (value < 0.2)
        {
            return "Detected movement from left to right.";
        }
        if (value >= 0.2 && value < 0.5)
        {
            return "Detected movement from right to left.";
        }
        else if (value >= 0.5 && value < 0.8)
        {
            return "Detected circle.";
        }
        else
        {
            return "Movement not detected.";
        }
    }
}