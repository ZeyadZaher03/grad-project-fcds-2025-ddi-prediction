package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"
)

type PubChemAutocompleteResp struct {
	Total           int `json:"total"`
	DictionaryTerms struct {
		Compound []string `json:"compound"`
	} `json:"dictionary_terms"`
	Status struct {
		Code int `json:"code"`
	} `json:"status"`
}

func PubChemAutocomplete(ctx context.Context, q string, limit int) ([]string, error) {
	u := fmt.Sprintf(
		"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/%s/json?limit=%d",
		url.PathEscape(q),
		limit,
	)

	req, _ := http.NewRequestWithContext(ctx, "GET", u, nil)
	client := &http.Client{Timeout: 10 * time.Second}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("pubchem autocomplete status=%d", resp.StatusCode)
	}

	var out PubChemAutocompleteResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}

	return out.DictionaryTerms.Compound, nil
}
