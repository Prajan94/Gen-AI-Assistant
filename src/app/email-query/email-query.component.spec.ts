import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EmailQueryComponent } from './email-query.component';

describe('EmailQueryComponent', () => {
  let component: EmailQueryComponent;
  let fixture: ComponentFixture<EmailQueryComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [EmailQueryComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(EmailQueryComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
