package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"
)

type DDIRequest struct {
	DrugAName string `json:"drugAName"`
	DrugBName string `json:"drugBName"`
}

type DDIResponse struct {
	DrugAName   string  `json:"drugAName"`
	DrugBName   string  `json:"drugBName"`
	SmilesA     string  `json:"smilesA"`
	SmilesB     string  `json:"smilesB"`
	Probability float64 `json:"probability"`
	Label       string  `json:"label"`
	Model       string  `json:"model"`
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func main() {
	inferenceURL := os.Getenv("INFERENCE_URL")
	if inferenceURL == "" {
		inferenceURL = "http://localhost:8001"
	}

	mux := http.NewServeMux()

	// Resolve SMILES endpoint
	mux.HandleFunc("/v1/resolve-smiles", func(w http.ResponseWriter, r *http.Request) {
		name := r.URL.Query().Get("name")
		if name == "" {
			http.Error(w, "missing name", 400)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
		defer cancel()

		smiles, err := ResolveSmilesPubChem(ctx, name)
		if err != nil {
			http.Error(w, "not found", 404)
			return
		}
		writeJSON(w, 200, map[string]string{
			"name":   name,
			"smiles": smiles,
		})
	})

	// Predict endpoint
	mux.HandleFunc("/v1/ddi/predict", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", 405)
			return
		}
		var req DDIRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid json", 400)
			return
		}
		if req.DrugAName == "" || req.DrugBName == "" {
			http.Error(w, "drug names required", 400)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 25*time.Second)
		defer cancel()

		// resolve to smiles
		smilesA, err := ResolveSmilesPubChem(ctx, req.DrugAName)
		if err != nil {
			log.Printf("resolve drugA %q failed: %v", req.DrugAName, err)
			http.Error(w, "drugA not found", 404)
			return
		}
		smilesB, err := ResolveSmilesPubChem(ctx, req.DrugBName)
		if err != nil {
			log.Printf("resolve drugB %q failed: %v", req.DrugBName, err)
			http.Error(w, "drugB not found", 404)
			return
		}

		// inference
		prob, severity, err := CallInference(ctx, inferenceURL, smilesA, smilesB)
		if err != nil {
			http.Error(w, "inference error", 502)
			return
		}

		writeJSON(w, 200, DDIResponse{
			DrugAName:   req.DrugAName,
			DrugBName:   req.DrugBName,
			SmilesA:     smilesA,
			SmilesB:     smilesB,
			Probability: prob,
			Label:       severity,
			Model:       "best_model",
		})
	})

	mux.HandleFunc("/v1/drugs/search", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query().Get("q")
		if q == "" {
			http.Error(w, "missing q", 400)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
		defer cancel()

		names, err := PubChemAutocomplete(ctx, q, 15)
		if err != nil {
			http.Error(w, "search failed", 502)
			return
		}

		writeJSON(w, 200, map[string]any{
			"query":   q,
			"results": names,
		})
	})

	addr := ":8080"
	log.Println("Go API listening on", addr, "INFERENCE_URL=", inferenceURL)
	log.Fatal(http.ListenAndServe(addr, mux))
}
