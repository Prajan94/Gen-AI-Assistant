import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TestCaseGeneratorAIComponent } from './test-case-generator-ai.component';

describe('TestCaseGeneratorAIComponent', () => {
  let component: TestCaseGeneratorAIComponent;
  let fixture: ComponentFixture<TestCaseGeneratorAIComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TestCaseGeneratorAIComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(TestCaseGeneratorAIComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
