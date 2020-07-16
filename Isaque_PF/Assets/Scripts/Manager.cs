using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;


public class Manager : MonoBehaviour
{
    [SerializeField]
    GameObject tile;
    [SerializeField]
    States states;
    [SerializeField]
    ExpectedOptions expected;
    [SerializeField]
    Button trainBtn;
    [SerializeField]
    MLP neuralNetwork;

    Vector3 startPos;
    Vector3 endPos;

    enum States
    {
        GeneratingDataset,
        TrainingNeuralNetwork,
        Validating
    };

    enum ExpectedOptions
    {
        LeftToRight,
        RightToLeft,
        Circle,
        NotRecognized
    };

    double currentNumHits;

    const int numCols = 21;
    const int numRows = 12;
    const string expectedLTR = "0.0", expectedRTL = "0.3", expectedC = "0.7", expectedNR = "1.0"; //LTR = Left to Right, RTL = Right to Left, C = Circle, NR = Not Recognized 

    void Start()
    {
        GenerateGrid();

        if (states == States.TrainingNeuralNetwork)
        {
            trainBtn.gameObject.SetActive(true);
        }
        else
        {
            trainBtn.gameObject.SetActive(false);
        }
    }

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            if (states == States.GeneratingDataset)
            {
                startPos = Camera.main.ScreenToViewportPoint(Input.mousePosition);
            }
        }

        if (Input.GetMouseButtonUp(0))
        {
            endPos = Camera.main.ScreenToViewportPoint(Input.mousePosition);
            
            if (states == States.GeneratingDataset)
            {
                WriteDataset(CheckMagnitudeString(), currentNumHits.ToString());
                ResetTiles();
            }
            if (states == States.Validating)
            {
                neuralNetwork.ValidateGesture(CheckMagnitudeDouble(), currentNumHits);
            }
        }

        RaycastHit hit;
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);

        if (Input.GetMouseButton(0))
        {
            if (states == States.GeneratingDataset)
            {
                if (Physics.Raycast(ray.origin, ray.direction, out hit, Mathf.Infinity))
                {
                    Tile tile = hit.transform.GetComponent<Tile>();
                    if (!tile.WasChecked)
                    {
                        tile.WasChecked = true;
                        ++currentNumHits;
                    }
                }
            }
        }
    }

    void GenerateGrid()
    {
        for (int i = 0; i < numCols; ++i)
        {
            for (int j = 0; j < numRows; ++j)
            {
                Instantiate(tile, new Vector3(i, -j, 0.0f), Quaternion.identity);
            }
        }
    }

    void ResetTiles()
    {
        Tile[] tiles = FindObjectsOfType<Tile>();

        foreach (Tile tile in tiles)
        {
            tile.WasChecked = false;
        }

        currentNumHits = 0.0;
    }

    string CheckMagnitudeString()
    {
        return (endPos - startPos).magnitude.ToString("F3").Replace(",", ".");
    }

    double CheckMagnitudeDouble()
    {
        return (double)(endPos - startPos).magnitude;
    }

    void WriteDataset(string distMagnitude, string numOfTiles)
    {
        string path = Application.dataPath + "\\../Dataset.txt";

        if (!File.Exists(path))
        {
            File.WriteAllText(path, "");
        }

        string content = "";

        if (expected == ExpectedOptions.LeftToRight)
        {
            content = $"{distMagnitude},{numOfTiles},{expectedLTR}\n";
        }
        else if (expected == ExpectedOptions.RightToLeft)
        {
            content = $"-{distMagnitude},{numOfTiles},{expectedRTL}\n";
        }
        else if (expected == ExpectedOptions.Circle)
        {
            content = $"{distMagnitude},{numOfTiles},{expectedC}\n";
        }
        else
        {
            content = $"{distMagnitude},{numOfTiles},{expectedNR}\n";
        }

        File.AppendAllText(path, content);
    }
}