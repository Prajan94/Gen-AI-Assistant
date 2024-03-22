import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PdfChatBotComponent } from './pdf-chat-bot.component';

describe('PdfChatBotComponent', () => {
  let component: PdfChatBotComponent;
  let fixture: ComponentFixture<PdfChatBotComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PdfChatBotComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(PdfChatBotComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
