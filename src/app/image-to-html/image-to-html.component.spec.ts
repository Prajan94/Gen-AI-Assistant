import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ImageToHtmlComponent } from './image-to-html.component';

describe('ImageToHtmlComponent', () => {
  let component: ImageToHtmlComponent;
  let fixture: ComponentFixture<ImageToHtmlComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ImageToHtmlComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ImageToHtmlComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
