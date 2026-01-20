package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type PredictReq struct {
	SmilesA string `json:"smilesA"`
	SmilesB string `json:"smilesB"`
}

type PredictResp struct {
	Probability float64 `json:"probability"`
	Severity    string  `json:"severity"`
}

func CallInference(ctx context.Context, inferenceURL string, smilesA, smilesB string) (float64, string, error) {
	body, _ := json.Marshal(PredictReq{SmilesA: smilesA, SmilesB: smilesB})

	req, _ := http.NewRequestWithContext(ctx, "POST", inferenceURL+"/predict", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return 0, "", fmt.Errorf("inference failed status=%d", resp.StatusCode)
	}

	var out PredictResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return 0, "", err
	}
	return out.Probability, out.Severity, nil
}
