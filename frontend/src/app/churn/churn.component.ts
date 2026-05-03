import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatDividerModule } from '@angular/material/divider';
import { MatIconModule } from '@angular/material/icon';
import { ChurnService, PredictionOutput } from './churn.service';

@Component({
  selector: 'app-churn',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatButtonModule,
    MatProgressSpinnerModule,
    MatDividerModule,
    MatIconModule
  ],
  templateUrl: './churn.component.html',
  styleUrls: ['./churn.component.css']
})
export class ChurnComponent implements OnInit {

  form!: FormGroup;
  result: PredictionOutput | null = null;
  loading = false;
  error = '';

  genderOptions = ['Male', 'Female'];
  yesNoOptions = ['Yes', 'No'];
  internetOptions = ['DSL', 'Fiber optic', 'No'];
  contractOptions = ['Month-to-month', 'One year', 'Two year'];
  paymentOptions = [
    'Electronic check',
    'Mailed check',
    'Bank transfer (automatic)',
    'Credit card (automatic)'
  ];
  multilineOptions = ['No', 'Yes', 'No phone service'];
  internetServiceOptions = ['No', 'Yes', 'No internet service'];

  constructor(
    private fb: FormBuilder,
    private churnService: ChurnService
  ) {}

  ngOnInit(): void {
    this.form = this.fb.group({
      gender:           ['Male', Validators.required],
      SeniorCitizen:    [0, Validators.required],
      Partner:          ['No', Validators.required],
      Dependents:       ['No', Validators.required],
      tenure:           [1, [Validators.required,
                             Validators.min(0),
                             Validators.max(72)]],
      PhoneService:     ['Yes', Validators.required],
      MultipleLines:    ['No', Validators.required],
      InternetService:  ['Fiber optic', Validators.required],
      OnlineSecurity:   ['No', Validators.required],
      OnlineBackup:     ['No', Validators.required],
      DeviceProtection: ['No', Validators.required],
      TechSupport:      ['No', Validators.required],
      StreamingTV:      ['No', Validators.required],
      StreamingMovies:  ['No', Validators.required],
      Contract:         ['Month-to-month', Validators.required],
      PaperlessBilling: ['Yes', Validators.required],
      PaymentMethod:    ['Electronic check', Validators.required],
      MonthlyCharges:   [29.85, [Validators.required, Validators.min(0)]],
      TotalCharges:     [29.85, [Validators.required, Validators.min(0)]]
    });
  }

  prefillHighRisk(): void {
    this.form.patchValue({
      gender: 'Male',
      SeniorCitizen: 0,
      Partner: 'No',
      Dependents: 'No',
      tenure: 2,
      PhoneService: 'Yes',
      MultipleLines: 'No',
      InternetService: 'Fiber optic',
      OnlineSecurity: 'No',
      OnlineBackup: 'No',
      DeviceProtection: 'No',
      TechSupport: 'No',
      StreamingTV: 'No',
      StreamingMovies: 'No',
      Contract: 'Month-to-month',
      PaperlessBilling: 'Yes',
      PaymentMethod: 'Electronic check',
      MonthlyCharges: 85.5,
      TotalCharges: 171.0
    });
  }

  prefillLowRisk(): void {
    this.form.patchValue({
      gender: 'Female',
      SeniorCitizen: 0,
      Partner: 'Yes',
      Dependents: 'Yes',
      tenure: 60,
      PhoneService: 'Yes',
      MultipleLines: 'Yes',
      InternetService: 'DSL',
      OnlineSecurity: 'Yes',
      OnlineBackup: 'Yes',
      DeviceProtection: 'Yes',
      TechSupport: 'Yes',
      StreamingTV: 'No',
      StreamingMovies: 'No',
      Contract: 'Two year',
      PaperlessBilling: 'No',
      PaymentMethod: 'Bank transfer (automatic)',
      MonthlyCharges: 45.0,
      TotalCharges: 2700.0
    });
  }

  onSubmit(): void {
    if (this.form.invalid) return;
    this.loading = true;
    this.error = '';
    this.result = null;

    this.churnService.predict(this.form.value).subscribe({
      next: (response: PredictionOutput) => {
        this.result = response;
        this.loading = false;
      },
      error: (err: any) => {
        this.error = 'Prediction failed. Make sure the API is running on port 8000.';
        this.loading = false;
        console.error(err);
      }
    });
  }

  getRiskColor(): string {
    if (!this.result) return '';
    switch (this.result.risk_level) {
      case 'High':   return '#e74c3c';
      case 'Medium': return '#f39c12';
      case 'Low':    return '#2ecc71';
      default:       return '#95a5a6';
    }
  }

  getRiskIcon(): string {
    if (!this.result) return '';
    switch (this.result.risk_level) {
      case 'High':   return 'warning';
      case 'Medium': return 'info';
      case 'Low':    return 'check_circle';
      default:       return 'help';
    }
  }
}