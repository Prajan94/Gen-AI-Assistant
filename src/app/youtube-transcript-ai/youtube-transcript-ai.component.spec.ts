import { ComponentFixture, TestBed } from '@angular/core/testing';

import { YoutubeTranscriptAIComponent } from './youtube-transcript-ai.component';

describe('YoutubeTranscriptAIComponent', () => {
  let component: YoutubeTranscriptAIComponent;
  let fixture: ComponentFixture<YoutubeTranscriptAIComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [YoutubeTranscriptAIComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(YoutubeTranscriptAIComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
