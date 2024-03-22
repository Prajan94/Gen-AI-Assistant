import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ServerLogAssistantComponent } from './server-log-assistant.component';

describe('ServerLogAssistantComponent', () => {
  let component: ServerLogAssistantComponent;
  let fixture: ComponentFixture<ServerLogAssistantComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ServerLogAssistantComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ServerLogAssistantComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
