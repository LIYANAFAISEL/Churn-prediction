import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface CustomerInput {
  gender: string;
  SeniorCitizen: number;
  Partner: string;
  Dependents: string;
  tenure: number;
  PhoneService: string;
  MultipleLines: string;
  InternetService: string;
  OnlineSecurity: string;
  OnlineBackup: string;
  DeviceProtection: string;
  TechSupport: string;
  StreamingTV: string;
  StreamingMovies: string;
  Contract: string;
  PaperlessBilling: string;
  PaymentMethod: string;
  MonthlyCharges: number;
  TotalCharges: number;
}

export interface FeatureFactor {
  feature: string;
  importance: number;
  importance_pct: string;
}

export interface PredictionOutput {
  churn_probability: number;
  churn_probability_pct: string;
  risk_level: string;
  prediction: string;
  top_factors: FeatureFactor[];
}

@Injectable({
  providedIn: 'root'
})
export class ChurnService {
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  predict(customer: CustomerInput): Observable<PredictionOutput> {
    return this.http.post<PredictionOutput>(
      `${this.apiUrl}/predict`,
      customer
    );
  }
}