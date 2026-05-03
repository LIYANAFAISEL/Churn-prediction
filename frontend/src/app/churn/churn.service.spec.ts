import { TestBed } from '@angular/core/testing';

import { ChurnService } from './churn.service';

describe('ChurnService', () => {
  let service: ChurnService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ChurnService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
