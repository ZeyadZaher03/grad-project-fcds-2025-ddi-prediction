package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"
)

type PubChemResp struct {
	PropertyTable struct {
		Properties []struct {
			CanonicalSMILES   string `json:"CanonicalSMILES"`
			IsomericSMILES    string `json:"IsomericSMILES"`
			SMILES            string `json:"SMILES"`
			ConnectivitySMILES string `json:"ConnectivitySMILES"`
		} `json:"Properties"`
	} `json:"PropertyTable"`
}

func ResolveSmilesPubChem(ctx context.Context, name string) (string, error) {
	u := fmt.Sprintf(
		"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/CanonicalSMILES/JSON",
		url.PathEscape(name),
	)

	req, _ := http.NewRequestWithContext(ctx, "GET", u, nil)
	client := &http.Client{Timeout: 10 * time.Second}

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode == 404 {
		return "", fmt.Errorf("not found")
	}
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("pubchem error status=%d", resp.StatusCode)
	}

	var out PubChemResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if len(out.PropertyTable.Properties) == 0 {
		return "", fmt.Errorf("no smiles")
	}

	prop := out.PropertyTable.Properties[0]
	smiles := prop.CanonicalSMILES
	if smiles == "" {
		smiles = prop.SMILES
	}
	if smiles == "" {
		smiles = prop.IsomericSMILES
	}
	if smiles == "" {
		smiles = prop.ConnectivitySMILES
	}
	if smiles == "" {
		return "", fmt.Errorf("empty smiles")
	}
	return smiles, nil
}
