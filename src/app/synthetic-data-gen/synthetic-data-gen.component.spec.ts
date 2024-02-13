import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SyntheticDataGenComponent } from './synthetic-data-gen.component';

describe('SyntheticDataGenComponent', () => {
  let component: SyntheticDataGenComponent;
  let fixture: ComponentFixture<SyntheticDataGenComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SyntheticDataGenComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(SyntheticDataGenComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
