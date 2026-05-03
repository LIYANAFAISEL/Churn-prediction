import { Component } from '@angular/core';
import { ChurnComponent } from './churn/churn.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [ChurnComponent],
  template: `<app-churn></app-churn>`
})
export class AppComponent {}