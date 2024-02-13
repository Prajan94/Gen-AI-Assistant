import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FigmaCodeGeneratorComponent } from './figma-code-generator.component';

describe('FigmaCodeGeneratorComponent', () => {
  let component: FigmaCodeGeneratorComponent;
  let fixture: ComponentFixture<FigmaCodeGeneratorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [FigmaCodeGeneratorComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(FigmaCodeGeneratorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
